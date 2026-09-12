"""Extract normalized attention weights from pre-trained MuLAN model for sequences in a file."""

import os
from argparse import ArgumentParser
from tqdm.auto import tqdm
# MPS: route ops without a Metal kernel to CPU rather than erroring; must precede `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import h5py  

import mulan


def get_args():
    parser = ArgumentParser(
        prog="mulan-att",
        description=__doc__,
    )
    parser.add_argument(
        "fasta_file",
        type=str,
        help="Fasta file containing sequences to extract attention weights."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mulan-ankh",
        help=f"Name of the pre-trained model. Must be one of: {mulan.get_available_models()}.",
    )
    parser.add_argument(
        "-o", "--output-file", type=str, default="attentions.h5", help="Output H5 file"
    )
    parser.add_argument(
        "-e",
        "--embeddings-dir",
        type=str,
        default=None,
        help="If not None, directory to store embeddings in PT format.",
    )
    parser.add_argument(
        "-m", "--no-minmax", action="store_true", help="If set, disables min-max scaling of attention weights."
    )
    args = parser.parse_args()
    if args.model_name not in mulan.get_available_models():
        raise ValueError(f"Invalid model name: {args.model_name}")
    return args


@torch.inference_mode()
def run(model_name, fasta_file, output_file, embeddings_dir=None, no_minmax=False):
    device = mulan.get_device()
    dataset = mulan.utils.parse_fasta(fasta_file)
    plm_name = model_name.split("-")[1]
    plm_model, plm_tokenizer = mulan.load_pretrained_plm(plm_name, device=device)
    model = mulan.load_pretrained(model_name)
    os.makedirs(os.path.dirname(output_file), exist_ok=True) if "/" in output_file else None
    if embeddings_dir is not None:
        os.makedirs(embeddings_dir, exist_ok=True)
    pbar = tqdm(initial=0, total=len(dataset), colour="red", dynamic_ncols=True, ascii="-#")
    pbar.set_description("Extracting attention weights")
    with h5py.File(output_file, "w") as f:
        for name, sequence in dataset.items():
            # embed sequence
            embedding = mulan.utils.embed_sequence(plm_model, plm_tokenizer, sequence)
            attention = model.encoder(embedding).attention
            attention = attention.squeeze(0).cpu().numpy().mean(axis=(-2, -3))
            if not no_minmax:
                attention = mulan.utils.minmax_scale(attention)
            else:
                attention = attention * len(sequence)  # rescale to an average of 1
            # save attention weights
            f.create_dataset(name, data=attention)
            if embeddings_dir is not None:
                mulan.utils.save_embedding(embedding, embeddings_dir, name)
            pbar.update(1)


def main():
    args = get_args()
    run(args.model_name, args.fasta_file, args.output_file, args.embeddings_dir, args.no_minmax)


if __name__ == "__main__":
    main()
    