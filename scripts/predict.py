"""Predict scores for given mutations in a protein complex."""

import os
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm
# MPS: route ops without a Metal kernel to CPU rather than erroring; must precede `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import pandas as pd

import mulan
import mulan.utils as utils


def get_args():
    parser = ArgumentParser(
        prog="mulan-predict",
        description=__doc__,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mulan-ankh",
        help=f"Name of the pre-trained model. Must be one of: {mulan.get_available_models()}.",
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="""
        Input file containing wild type sequences and mutations to be scored.
        Each line must contain columns with  the name of the complex, the interacting 
        sequences and the mutations to be scored, separated by a comma, in the format 
        <wt_aa><chain:A,B><position><mut_aa>.
        Multiple-point mutations can be provided, separated by a column.
        Example: 'C1 SEQ1 SEQ2 AA1G:AA2T,CA3C:QB4A' """,
    )
    parser.add_argument(
        "-s",
        "--scores-file",
        type=str,
        default=None,
        help="""
        TXT File containing zero-shot scores for imulan model. Each line must contain 
        the name of the complex, the correesponding mutation with the same format as in 
        'input_file' and the score, separated by a white spaces.""",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.txt",
        help="Output file. Set to 'output.txt' by default.",
    )
    parser.add_argument(
        "--store-embeddings",
        action="store_true",
        help="Store embeddings in PT format. Output directory is the same of 'output_file'.",
    )
    args = parser.parse_args()
    if args.model_name not in mulan.get_available_models():
        raise ValueError(f"Invalid model name: {args.model_name}")
    return args


def parse_input(input_file):
    """Parse input file and return a list of records.

    One record per line, four whitespace-separated fields::

        <complex>  <seqA>  <seqB>  <mut1,mut2,...>

    This is **not** a FASTA. The unqualified ``split()`` used to raise
    ``ValueError: not enough values to unpack`` on the first line of one, which named neither
    the file nor the expected format -- and the docs pointed at ``examples/example.fasta``, so
    the very first command a new user ran produced exactly that error.
    """
    data = []
    with open(input_file) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 4:
                hint = ""
                if line.startswith(">"):
                    hint = (" This looks like a FASTA record; predict takes a table, not a "
                            "FASTA. See examples/sample_mut.txt.")
                raise ValueError(
                    f"{input_file}:{lineno}: expected 4 whitespace-separated fields "
                    f"(complex, seqA, seqB, mutations) but found {len(fields)}.{hint}"
                )
            complex_name, seq1, seq2, mutations = fields
            mutations = [tuple(m.split(":")) for m in mutations.split(",")]
            data.append(
                {
                    "complex": complex_name,
                    "sequences": (seq1, seq2),
                    "mutations": mutations,
                }
            )
    if not data:
        raise ValueError(f"{input_file} contained no usable records.")
    return data


def parse_zs_scores(scores_file):
    """Parse zero-shot scores file and return a dictionary."""
    zs_scores = defaultdict(dict)
    with open(scores_file) as f:
        for line in f:
            # print(line)
            complex_name, mutations, score = line.strip().split()
            # A multi-point mutation group is colon-separated here to match parse_input
            # (`tuple(m.split(":"))`); splitting on "," instead produced a single-element
            # tuple that never matched the loop's key, so zs_scores.get() returned None
            # for every multi-point mutation.
            mutations = tuple(mutations.split(":"))
            # print(line)
            # print(complex_name, mutations, score)
            zs_scores[complex_name].update(
                {mutations: torch.tensor(float(score), dtype=torch.float32).unsqueeze(0)}
            )
    return zs_scores


@torch.inference_mode()
def run(model_name, input_file, scores_file, output_file, store_embeddings):
    device = mulan.get_device()
    data = parse_input(input_file)
    num_iterations = sum(len(d["mutations"]) for d in data)
    scores = []
    # Load the checkpoint first. This fork does not redistribute them, so a missing checkpoint
    # is the common case -- and loading the PLM first meant downloading several GB before the
    # failure a reader was always going to hit.
    model = mulan.load_pretrained(model_name)
    plm_name = model_name.split("-")[1]
    plm_model, plm_tokenizer = mulan.load_pretrained_plm(plm_name, device=device)
    model = model.eval()
    if "imulan" in model_name and scores_file is None:
        raise ValueError("Zero-shot scores file must be provided for imulan models.")
    zs_scores = parse_zs_scores(scores_file) if scores_file is not None else {}
    if store_embeddings:
        # `or "."` is load-bearing: the default --output-file is a bare "output.txt", whose
        # dirname is "", and os.makedirs("") raises FileNotFoundError. So --store-embeddings
        # crashed unless you happened to pass an -o containing a directory.
        embeddings_dir = os.path.dirname(output_file) or "."
        os.makedirs(embeddings_dir, exist_ok=True)

    pbar = tqdm(initial=0, total=num_iterations, colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Running prediction")
    for complex in data:
        # compute embeddings for wt sequences
        complex_name = complex["complex"]
        seq1, seq2 = complex["sequences"]
        seq1_embedding = utils.embed_sequence(plm_model, plm_tokenizer, seq1)
        seq2_embedding = utils.embed_sequence(plm_model, plm_tokenizer, seq2)
        for mutations in complex["mutations"]:
            mut_seq1, mut_seq2 = utils.parse_mutations(mutations, seq1, seq2)
            if mut_seq1 != seq1:
                mut_seq1_embedding = utils.embed_sequence(plm_model, plm_tokenizer, mut_seq1)
            else:
                mut_seq1_embedding = seq1_embedding
            if mut_seq2 != seq2:
                mut_seq2_embedding = utils.embed_sequence(plm_model, plm_tokenizer, mut_seq2)
            else:
                mut_seq2_embedding = seq2_embedding
            inputs = [seq1_embedding, seq2_embedding, mut_seq1_embedding, mut_seq2_embedding]
            zs_score = zs_scores.get(complex_name, {}).get(mutations, None)
            score = (
                model(
                    inputs_embeds=inputs,
                    zs_scores=zs_score.to(device) if zs_score is not None else None,
                )
                .squeeze()
                .item()
            )
            scores.append(
                {"complex": complex_name, "mutations": ":".join(mutations), "score": score}
            )

            if store_embeddings:
                utils.save_embedding(
                    mut_seq1_embedding, 
                    embeddings_dir, 
                    f"{complex_name}_{'-'.join([mut for mut in mutations if mut[1] == 'A'])}"
                )
                utils.save_embedding(
                    mut_seq2_embedding, 
                    embeddings_dir, 
                    f"{complex_name}_{'-'.join([mut for mut in mutations if mut[1] == 'B'])}"
                )
            pbar.update(1)
            
        if store_embeddings:
            utils.save_embedding(seq1_embedding, embeddings_dir, f"{complex_name}_A")
            utils.save_embedding(seq2_embedding, embeddings_dir, f"{complex_name}_B")
                
    df = pd.DataFrame.from_records(scores)
    df.to_csv(output_file, index=False, sep="\t", float_format="%.3f")


def main():
    args = get_args()
    run(
        args.model_name, args.input_file, args.scores_file, args.output_file, args.store_embeddings
    )


if __name__ == "__main__":
    main()
