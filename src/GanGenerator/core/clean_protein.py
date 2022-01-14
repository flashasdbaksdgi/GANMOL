import Bio
import click
from Bio.PDB import PDBParser, PDBIO, Select


"""Remove HETATM from PDB file
    """


class RemoveHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if Bio.PDB.Polypeptide.is_aa(residue, standard=True) else 0


class RemoveChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return 1 if chain.get_id() == self.chain_id else 0


def clean_protein(input_file, output_file, chain_id=None, verbose=False):
    pdb = PDBParser(QUIET=True).get_structure("protein", input_file)
    io = PDBIO()
    io.set_structure(pdb)
    if chain_id:
        io.save(output_file, RemoveChainSelect(chain_id))
        if verbose:
            print(f"Protein chain {chain_id} save at {output_file}")
    else:
        io.save(output_file, RemoveHetSelect())
        if verbose:
            print(f"Protein save at {output_file}")


@click.command()
@click.option(
    "--input_file", type=click.File("r"), help="Input PDB file", required=True
)
@click.option(
    "--output_file", type=click.File("w"), help="Output filename", required=True
)
@click.option("--chain_id", type=str, help="Chain ID", required=False)
@click.option("--verbose", is_flag=True, help="Verbose mode", default=True)
def main(input_file, output_file, chain_id, verbose):
    clean_protein(input_file, output_file, chain_id=chain_id, verbose=verbose)


if __name__ == "__main__":
    """Remove HETATM from PDB file"""
    main()
