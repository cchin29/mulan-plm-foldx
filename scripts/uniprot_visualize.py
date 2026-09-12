import argparse
import requests
import subprocess
import time
from pathlib import Path

import h5py


AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v6.pdb"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{}.fasta"


class PymolSession:
    """Manages a PyMOL subprocess session."""

    def __init__(self):
        cmd = ["pymol", "-p"]  # GUI mode
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
        time.sleep(2)  # Wait for PyMOL to start

    def __call__(self, command: str) -> None:
        """Sends a command to the PyMOL session."""
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def close(self) -> None:
        """Closes the PyMOL session safely."""
        self.process.stdin.close()
        self.process.wait()
        self.process.terminate()


def get_args():
    parser = argparse.ArgumentParser(
        description="Plot MuLAN attention scores on AlphaFold predicted structure using PyMOL"
    )
    parser.add_argument("uniprot_id", type=str, help="UniProt ID of the protein")
    # parser.add_argument("attention_file", type=str, help="Path to the attention file in HDF5 format")
    parser.add_argument(
        "-p",
        "--pdb-file",
        type=Path,
        default=None,
        help=(
            "Path to the directory PDB file. If not found, the script will try to fetch the"
            " corresponding file from the AlphaFold database. Default is ./pdbs/<uniprot_id>.pdb"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the output directory. Default is ./outputs",
    )
    args = parser.parse_args()
    if args.pdb_file is None:
        args.pdb_file = Path("pdbs") / f"{args.uniprot_id}.pdb"
    return args


def get_scores(uniprot_id, output_dir):
    # TODO: modify by fetch and store precomputed scores (format can also be changed)
    # with h5py.File(Path(__file__).parent / "attentions.h5", "r") as f:  
    # run mulan-att script if scores are not found
    try:
        # fetch protein fasta file from uniprot
        response = requests.get(UNIPROT_URL.format(uniprot_id))
        response.raise_for_status()
        # save to file
        with open(output_dir / f"{uniprot_id}.fasta", "w") as f:
            f.write(response.text)
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch fasta sequence for {uniprot_id}: {e}")

    with h5py.File(Path(output_dir) / "attentions.h5", "r") as f:
        if uniprot_id in f:
            return f[uniprot_id][:]
        else:
            print(f"Scores for {uniprot_id} not found in attentions.h5, computing using mulan-att.")
            try:
                # run mulan-att
                subprocess.run(
                    [
                        "mulan-att",
                        output_dir / f"{uniprot_id}.fasta",
                        "-o",
                        output_dir / f"{uniprot_id}.h5",
                    ]
                )
                with h5py.File(output_dir / f"{uniprot_id}.h5", "r") as f:
                    return f["sp"][:]
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Failed to run mulan-att: {e}")


def plot_on_structure(pdb_file, array, manual_offset=1):
    array = [float(x) for x in array]
    min_score = min(array)
    max_score = max(array)

    # Start PyMOL session
    pms = PymolSession()
    mol = pdb_file.stem
    cmap = "blue_white_red"

    def _plot_on_structure():
        pms(f"load {pdb_file}")
        pms("hide everything")
        pms("show cartoon, chain A")
        pms("center chain A")
        pms("set ray_trace_color, black")

        # Set scores as B-factors and color by spectrum
        pms(f"alter {mol}, b=-1")
        pms(
            f"for res_id, score in enumerate({array}): cmd.alter(f'resi {{res_id +"
            f" {manual_offset}}} and chain A', f'b={{score}}')"
        )
        time.sleep(3)  # Wait for the scores to be set

        pms(f"spectrum b, {cmap}, minimum={min_score}, maximum={max_score}")

        # Add colorbar
        pms(f"ramp_new colorbar, {mol}, [{min_score}, {max_score}], {cmap.split('_')}")

    try:
        _plot_on_structure()
    except Exception as e:
        pms.close()
        raise e

    try:
        print("PyMOL session is running. Press Ctrl+C to close.")
        pms.process.wait()  # Keep PyMOL open
    except KeyboardInterrupt:
        print("Closing PyMOL session.")
        pms.close()


def fetch_afdb(uniprot_id, pdb_file):
    if not pdb_file.exists():
        pdb_file.parent.mkdir(parents=True, exist_ok=True)
        url = AF_URL.format(uniprot_id)
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(pdb_file, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {url} to {pdb_file}")
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch {url}: {e}")


if __name__ == "__main__":
    args = get_args()
    fetch_afdb(args.uniprot_id, args.pdb_file)
    array = get_scores(args.uniprot_id, args.output_dir)
    plot_on_structure(args.pdb_file, array)
