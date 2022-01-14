import fpocket
import subprocess
from rich.console import Console

console = Console()


def fpocket_predict(protein):
    """
    Test the fpocket module.
    """
    command = f"fpocket -f {protein} "
    out_log = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    out_log = out_log.stdout.read()
    out_log = out_log.decode("utf-8")
    console.print(out_log)


if __name__ == "__main__":
    # import os
    # import sys
    # import pytest

    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # pytest.main(
    #    [
    #        "-s",
    #        "/mnt/c/Users/Takshan/Desktop/PhD/LOCAL/lab09/DEV/GAN/notebooks/grid/6nzp_protein.pdb",
    #    ]
    # )
    fpocket(
        "/mnt/c/Users/Takshan/Desktop/PhD/LOCAL/lab09/DEV/GAN/notebooks/grid/6nzp_protein.pdb"
    )
