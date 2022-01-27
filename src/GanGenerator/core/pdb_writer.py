# Formated string for ATOM lines in PDB files

# field id	definition	length	format	range	string slicing(Python)
# 1	"ATOM " or "HETATM"	6	{: 6s}	01-06	[0:6]
# 2	atom serial number	5	{: 5d}	07-11	[6:11]
# 3	atom name	4	{: ^ 4s}	13-16	[12:16]
# 4	alternate location indicator	1	{: 1s}	17	[16:17]
# 5	residue name	3	{: 3s}	18-20	[17:20]
# 6	chain identifier	1	{: 1s}	22	[21:22]
# 7	residue sequence number	4	{: 4d}	23-26	[22:26]
# 8	code for insertion of residues	1	{: 1s}	27	[26:27]
# 9	orthogonal coordinates for X ( in Angstroms)	8	{: 8.3f}	31-38	[30:38]
# 10	orthogonal coordinates for Y ( in Angstroms)	8	{: 8.3f}	39-46	[38:46]
# 11	orthogonal coordinates for Z ( in Angstroms)	8	{: 8.3f}	47-54	[46:54]
# 12	occupancy	6	{: 6.2f}	55-60	[54:60]
# 13	temperature factor	6	{: 6.2f}	61-66	[60:66]
# 14	element symbol	2	{: > 2s}	77-78	[76:78]
# 15	charge on the atom	2	{: 2s}	79-80	[78:80]

# pdb_line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}  {:>2s}{:2s}".format(
# ...)
#import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

ATOM_TYPES = {
    0: "C",
    1: "H",
    2: "N",
    3: "O",

}

ATOM_TYPES_INV = {v: k for k, v in ATOM_TYPES.items()}


def sample_generator(sample_mol, atom_types):

    molecule = Chem.MolFromPDBFile(sample_mol, removeHs=False)
    MOL_SAMPLE = torch.zeros(len(atom_types), molecule.GetNumAtoms(), 3)

    for i in range(len(atom_types)):
        for atom, idx in zip(molecule.GetAtoms(), range(len(MOL_SAMPLE[i]))):
            symbol = atom.GetSymbol()
            channel = atom_types[symbol]
            if channel == i:
                coord = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
                MOL_SAMPLE[i][idx] += torch.tensor(
                    [coord.x, coord.y, coord.z], dtype=torch.float32)
    print(MOL_SAMPLE)
    return MOL_SAMPLE


def GetAtomPosition_AtomSymbol(mol):
    "Get atom position and symbol from molecule"
    for atom in mol.GetAtoms():
        coord = atom.GetConformer().GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()


def pdb_writer(mol: torch.tensor, atom_type: dict, **kwargs) -> None:

    if mol.size()[0] != len(atom_type):
        raise ValueError(
            "ATOM_TYPES and mol(Channnel) must have the same length")
    SAMPLE = kwargs.get("SAMPLE", "")
    mol_type = kwargs.get("atom_type", "ligand")
    atom_serial_number = kwargs.get("atom_serial_number", 0)
    atom_name = kwargs.get("atom_name", "    ")
    alternate_location_indicator = kwargs.get(
        "alternate_location_indicator", "")
    residue_name = kwargs.get("residue_name", "   ")
    chain_identifier = kwargs.get("chain_identifier", "")
    residue_sequence_number = kwargs.get("residue_sequence_number", "")
    insertion_of_residues = kwargs.get(
        "insertion_of_residues", "")
    occupancy = kwargs.get("occupancy", 1.0000)
    temperature_factor = kwargs.get("temperature_factor", 0.0000)
    element_symbol = kwargs.get("element_symbol", "  ")
    charge_on_the_atom = kwargs.get("charge_on_the_atom", "  ")

    if mol_type == "ligand":
        molecule = "HETATM"
        residue_name = "UNK "
        chain_identifier = " "
        residue_sequence_number = 0

# pdb_line = f"{molecule:6s}{atom_serial_number:5d} {atom_name:^4s}{alternate_location_indicator:1s}{residue_name:3s} {chain_identifier:1s}{residue_sequence_number:4d}{insertion_of_residues:1s}{coordinates_X:8.3f}{coordinates_Y:8.3f}{coordinates_Z:8.3f}{occupancy:6.2f}{temperature_factor:6.2f}  {element_symbol:>2s}{charge_on_the_atom:2s}"

# print(pdb_line)
    mol_list = []
    for i in range(mol.size()[0]):
        for j in range(mol.size()[1]):
            if mol[i][j].sum() != 0.00:
                atom_serial_number += 1
            atom_name = atom_type[i]
            coordinates_X = mol[i][j][0]
            coordinates_Y = mol[i][j][1]
            coordinates_Z = mol[i][j][2]
            element_symbol = atom_type[i]
            #pdb_line = f"{molecule:6s}{atom_serial_number:5d} {atom_name:^4s}{alternate_location_indicator:1s}{residue_name:3s} {chain_identifier:1s}{residue_sequence_number:4d}{insertion_of_residues:1s}{coordinates_X:8.3f}{coordinates_Y:8.3f}{coordinates_Z:8.3f}{occupancy:6.2f}{temperature_factor:6.2f}  {element_symbol:>2s}{charge_on_the_atom:2s}"
            pdb_line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format(
                molecule, atom_serial_number, atom_name, alternate_location_indicator, residue_name, chain_identifier, residue_sequence_number, insertion_of_residues, coordinates_X, coordinates_Y, coordinates_Z, occupancy, temperature_factor, element_symbol, charge_on_the_atom)
            print(pdb_line)
            mol_list.append(pdb_line)
    # mol_list.insert(0, "") # insert header here
    with open("1Tensor_test.pdb", "w+") as fw:
        for i in mol_list:
            try:
                # print(i)
                if float(i[30:38]) == 0.000 and float(i[39:46]) == 0.000:
                    pass
                else:
                    print(i, file=fw)
            except ValueError:
                print(i, file=fw)

    chem_mol = Chem.MolFromPDBFile(
        "1Tensor_test.pdb", removeHs=False)
    # print(Chem.MolToPDBBlock(chem_mol))
    print("==========================")
    #chem_mol = Chem.AddHs(chem_mol)
    # AllChem.MMFFOptimizeMolecule(chem_mol)
    Chem.MolToPDBFile(chem_mol, "3Tensor_test.pdb")


def main():

    SAMPLE_MOL = '/media/takshan/F6FCFB6BFCFB2511/Users/Takshan/Desktop/PhD/LOCAL/lab09/DEV/GAN/data/processed/4wvs/test.pdb'

    # sample_mol = torch.rand(4, 10, 3)*50-20
    MOL_SAMPLE = sample_generator(SAMPLE_MOL, ATOM_TYPES_INV)
    pdb_writer(MOL_SAMPLE,  ATOM_TYPES, SAMPLE=SAMPLE_MOL)


if __name__ == "__main__":
    main()
