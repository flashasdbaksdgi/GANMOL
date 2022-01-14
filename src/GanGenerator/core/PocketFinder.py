import subprocess
import os
from glob import glob
from rich.console import Console
import pandas as pd
import gzip

console = Console()


class PocketFinder:
    """This class is used to run p2rank on a protein pdb file and generate ranked pockets"""

    def __init__(self, BASE_DIR, **kwargs):
        self.BASE_DIR = BASE_DIR
        self.verbose = kwargs.get("verbose", 3)
        self.PRANK = kwargs.get(
            "p2rank_path",
            "/mnt/c/Users/Takshan/Desktop/PhD/LOCAL/lab09/DEV/GAN/TEST/p2rank/distro/prank",
        )
        if self.verbose > 0:
            console.print(
                f"Base Dir for Pocket Finder set as {BASE_DIR}", style="bold red"
            )
        if self.verbose > 1:
            console.print(f"P2rank set as {self.PRANK}", style="bold green")
            console.print("Use 'p2rank_path=/some/path/to/p2rank' to set the path")

    def p2rank_predict(
        self,
        protein_file,
        output_dir=None,
        p2rank_path=None,
        threads=4,
        back=False,
        verbose=True,
    ):
        """This function runs p2rank on a protein pdb file and generates ranked pockets"""
        if p2rank_path is None:
            PRANK = self.PRANK
        else:
            PRANK = p2rank_path

        if output_dir is None:
            output_dir = (
                self.BASE_DIR + os.path.basename(protein_file).split(".")[0] + "_p2rank"
            )
        # Run p2rank pocket
        command = (
            f"{PRANK} predict -f {protein_file} -o {output_dir} -threads {threads}"
        )
        run_log = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )  # startup JVM is slow so we need to run it in the background. NO use of stdout
        if run_log and verbose:
            for line in run_log.stderr.decode("utf-8").split("\n"):
                # if "Error" or "ERROR" in line:
                print(line)
                #    raise Exception(line)
            for line in run_log.stdout.decode("utf-8").split("\n"):
                print(line)
        # Return the output
        self.PROT_FILE = os.path.basename(protein_file)
        self.OUT_DIR = output_dir
        if back:
            return run_log.stdout.decode("utf-8")

    def p2rank_result_parser(self, molecule=None, dir=None):
        """This function parses the p2rank output and returns the ranked pockets"""
        # Read the p2rank output
        if dir is None:
            try:
                dir = self.OUT_DIR if self.OUT_DIR is not None else None
            except AttributeError:
                raise Exception("No output directory provided")

        if molecule is None:
            try:
                molecule = self.PROT_FILE
            except AttributeError:
                console.print("No molecule file provided")
        else:
            try:
                molecule_exists = os.path.join(dir, molecule + ".pdb_predictions.csv")
                molecule = os.path.basename(molecule_exists)
            except FileNotFoundError:
                raise Exception(f"{molecule} file does not exist in the directory")

        if not molecule.endswith(".pdb_predictions.csv"):
            molecule = molecule + ".pdb_predictions.csv"

        predictions = pd.read_csv(os.path.join(dir, molecule), header=0)
        predictions.columns = predictions.columns.str.strip()
        # Get the ranked pockets
        self.PREDICTED_POCKETS = predictions
        return predictions

    def PocketPoint(self):
        """This function extracts the point cloud from a pocket"""
        generated_location = ["visualizations", "data"]
        gz_file = os.path.join(
            self.OUT_DIR,
            os.path.join(
                os.path.join(*generated_location),
                f"{self.PROT_FILE.split('.pdb')[0]}.pdb_points.pdb.gz",
            ),
        )
        with gzip.open(gz_file, "rb") as f:
            file_content = f.read()
            point_clouds = file_content.decode("utf-8")

        POINT_XYZ = []
        for line in point_clouds.split("\n"):
            if line.startswith("HETATM"):
                # print(line[25:26])
                if line[25:26] == "1":
                    _ = [float(x) for x in line.split()[6:9]]
                    POINT_XYZ.append(_)
        return POINT_XYZ


if __name__ == "__main__":
    """Predict pocket residues using p2rank"""
    print("This is a module for Pocket Finder for SSU")
