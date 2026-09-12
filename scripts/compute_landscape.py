"""Compute full mutational landscape of a protein in a given complex."""

import os
import re
from argparse import ArgumentParser
from tqdm.auto import tqdm
import pandas as pd
# MPS: route ops without a Metal kernel to CPU rather than erroring; must precede `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import numpy as np

import mulan
import mulan.utils as utils


def get_args():
    parser = ArgumentParser(
        prog="mulan-landscape",
        description=__doc__,
    )
    parser.add_argument(
        "sequences",
        type=str,
        help="""Sequences strings, separated by column character. 
        The first is the sequence to be scored, the other is the partner.""",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="mulan-ankh",
        help=f"""Name of the pre-trained model. Must be one of: {mulan.get_available_models()}.
        If the model is a version of imulan, a TXT file containing zero-shot scores 
        must be provided, where lines correspond to mutated positions and columns 
        to amino acids, in alphabetic order for the single letter name, separated 
        by spaces.""",
    )
    parser.add_argument(
        "-s",
        "--scores-file",
        type=str,
        default=None,
        help="TXT File containing zero-shot scores for imulan model.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="output",
        help="Output directory. Set to 'output' by default.",
    )
    parser.add_argument(
        "--no-ranksort", action="store_true", help="Do not ranksort computed scores."
    )
    parser.add_argument(
        "-e",
        "--embeddings-dir",
        type=str,
        default=None,
        help="If not None, directory to store embeddings in PT format.",
    )
    args = parser.parse_args()
    if args.model_name not in mulan.get_available_models():
        raise ValueError(f"Invalid model name: {args.model_name}")
    args.sequences = args.sequences.upper()
    if "imulan" in args.model_name and args.scores_file is None:
        raise ValueError("Zero-shot scores file must be provided for imulan models.")
    return args


@torch.inference_mode()
def score_mutation(
    model,
    plm_model,
    plm_tokenizer,
    wildtype_embedding,
    partner_embedding,
    mut_name,
    mutation,
    zs_score,
    embeddings_dir,
):
    """Score a mutation in a protein sequence."""
    mutation_embedding = utils.embed_sequence(plm_model, plm_tokenizer, mutation)
    score = (
        model(
            inputs_embeds=[
                wildtype_embedding,
                partner_embedding,
                mutation_embedding,
                partner_embedding,
            ],
            zs_scores=zs_score,
        )
        .squeeze()
        .item()
    )
    if embeddings_dir is not None:
        utils.save_embedding(mutation_embedding, embeddings_dir, mut_name)
    return score


def run(
    model_name, sequences, output_dir, scores_file=None, ranksort_output=True, embeddings_dir=None
):
    """Compute full mutational landscape of a protein in a given complex."""
    device = mulan.get_device()
    seq1, seq2 = re.sub(r"[UZOB]", "X", sequences).split(":")
    output = torch.zeros(len(seq1), len(utils.AAs))
    os.makedirs(output_dir, exist_ok=True)
    if embeddings_dir is not None:
        os.makedirs(embeddings_dir, exist_ok=True)

    # load models
    plm_name = model_name.split("-")[1]
    plm_model, plm_tokenizer = utils.load_pretrained_plm(plm_name, device=device)
    model = mulan.load_pretrained(model_name)
    model.eval()

    if scores_file is not None:
        zs_scores = np.loadtxt(scores_file)
        zs_scores = torch.tensor(zs_scores, dtype=torch.float32)

    # embed wildtype sequences
    wildtype_embedding = utils.embed_sequence(plm_model, plm_tokenizer, seq1)
    partner_embedding = utils.embed_sequence(plm_model, plm_tokenizer, seq2)
    if embeddings_dir is not None:
        utils.save_embedding(wildtype_embedding, embeddings_dir, "WT")
        utils.save_embedding(partner_embedding, embeddings_dir, "PARTNER")

    # iterate over single-point mutations
    num_iterations = len(seq1) * (len(utils.AAs) - 1)
    pbar = tqdm(initial=0, total=num_iterations, colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Scoring mutations")
    for mut_name, mutation in utils.mutation_generator(seq1):
        i, aa = int(mut_name[1:-1]) - 1, mut_name[-1]
        zs_score = None if scores_file is None else zs_scores[i, utils.aa2idx[aa]].to(device)
        score = score_mutation(
            model,
            plm_model,
            plm_tokenizer,
            wildtype_embedding,
            partner_embedding,
            mut_name,
            mutation,
            zs_score,
            embeddings_dir,
        )
        output[i, utils.aa2idx[aa]] = score
        pbar.update(1)

    # save output
    output = output.cpu().numpy()
    if ranksort_output:
        wt_index = (range(len(seq1)), tuple([utils.aa2idx[aa] for aa in seq1]))
        output[wt_index] = -10000  # set to very low value to give lowest scores to wt residues
        output = utils.ranksort(output)
    output = pd.DataFrame(
        output, columns=utils.AAs, index=[f"{i+1}{aa}" for i, aa in enumerate(seq1)]
    )
    output.to_csv(os.path.join(output_dir, "landscape.csv"), float_format="%.3f")


def main():
    args = get_args()
    run(
        args.model_name,
        args.sequences,
        args.output_dir,
        args.scores_file,
        not args.no_ranksort,
        args.embeddings_dir,
    )


if __name__ == "__main__":
    main()
