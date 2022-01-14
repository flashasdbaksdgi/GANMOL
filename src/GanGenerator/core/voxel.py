# Takes PDB file and outputs full maps for different atomtypes
# Uses GPU for voxelation

import os

import numpy as np  # CPU

# import cupy as cp #FOR CUDA
import time
import torch as torch
import pickle
from Bio.PDB.PDBParser import PDBParser

torch.cuda.is_available()

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


parser = PDBParser(PERMISSIVE=1)

# Settings:-------------------------------------------------------------------

# Defaults:
CUTOFF = 10  # CUTOFF (beyond which a point cannot feel an atom) (angstroms)
STD = 1.0  # standard deviation for gaussian
V_SIZE = 0.5  # voxel size
PAD = 20  # PAD size around edges of protein in angstroms
N_ATOMTYPES = 4  # Number of types of atoms, must be aligned with atom_types.py
proc_file = 0  # number of files processed

# Identify file locations:
curr_path = (
    "/mnt/c/Users/Takshan/Desktop/PhD/LOCAL/lab09/DEV/GAN/notebooks/grid"  # os.getcwd()
)

pdb_path = os.path.join(curr_path, "chainA-clean.pdb")
pickle_path = os.path.join(curr_path, "3pickle_perpdb")
all_files = os.listdir(curr_path)


def atom_id(atom):
    N_ATOMTYPES = 4
    id_mat = torch.zeros([1, N_ATOMTYPES])[0]
    # Atom type 1: Carbon
    if atom.get_name()[0] == "C":
        id_mat[0] = 1

    # Atom type 2: Nitrogen
    if atom.get_name()[0] == "N":
        id_mat[1] = 1

    # Atom type 3: Oxygen
    if atom.get_name()[0] == "O":
        id_mat[2] = 1

    # Atom type 4: Sulfur
    if atom.get_name()[0] == "S":
        id_mat[3] = 1

    return id_mat


# Calculate density felt at a point from a distance from an atom center:
def atom_density(distance, STD):
    density = torch.exp(-(distance ** 2) / (2 * STD ** 2))
    return density


# Calculate center of geometry for a given residue:
def res_cog(residue):
    coord = [
        residue.get_list()[i].get_coord()
        for i in range(0, torch.shape(residue.get_list())[0])
    ]
    cog = torch.mean(coord, axis=0)
    return


start = time.time()


# Load in PDB files:-----------------------------------------------------------

for item in all_files:
    file, extension = os.path.splitext(item)
    if extension == ".pdb":
        proc_file += 1

        print("Processing File", proc_file, file)

        structure_id = file
        filename = os.path.join(curr_path, item)
        structure = parser.get_structure(structure_id, filename)

        ## Populate a grid with atomic densities:--------------------------------------
        # Define grid edges
        coord_list = [atom.coord for atom in structure.get_atoms()]
        # print(coord_list)
        # print(np.shape(coord_list)[0])
        var = torch.tensor(np.array(coord_list), dtype=torch.float32)
        _shape = var.shape[0]
        # print(_shape)

        xmin = min([coord_list[i][0] for i in range(0, _shape)])
        xmin = xmin - PAD
        xmax = max([coord_list[i][0] for i in range(0, _shape)])
        xmax = xmax + PAD

        ymin = min([coord_list[i][1] for i in range(0, _shape)])
        ymin = ymin - PAD
        ymax = max([coord_list[i][1] for i in range(0, _shape)])
        ymax = ymax + PAD

        zmin = min([coord_list[i][2] for i in range(0, _shape)])
        zmin = zmin - PAD
        zmax = max([coord_list[i][2] for i in range(0, _shape)])
        zmax = zmax + PAD

        linx = torch.arange(xmin, xmax, V_SIZE)
        liny = torch.arange(ymin, ymax, V_SIZE)
        linz = torch.arange(zmin, zmax, V_SIZE)

        gridx, gridy, gridz = torch.meshgrid(linx, liny, linz, indexing="ij")
        # gridshape = torch.shape(gridx)
        gridshape = gridx.shape

        # Fill densities into grid

        # occupancy = torch.zeros(
        #    [N_ATOMTYPES, torch.shape(linx)[0], cp.shape(liny)[0], cp.shape(linz)[0]]
        # )
        occupancy = torch.zeros(
            [N_ATOMTYPES, (linx.shape)[0], (liny.shape)[0], (linz.shape)[0]]
        )

        for residue in list(structure.get_residues()):
            print("Working on Residue", residue)
            for atom in residue.get_list():
                id_mat = atom_id(atom)
                for i in range(0, N_ATOMTYPES):
                    if id_mat[i] == 1:
                        atomcoord = atom.get_coord()
                        for x in torch.where(abs(linx - atomcoord[0]) < CUTOFF / 2.0)[
                            0
                        ]:
                            for y in torch.where(
                                abs(liny - atomcoord[1]) < CUTOFF / 2.0
                            )[0]:
                                for z in torch.where(
                                    abs(linz - atomcoord[2]) < CUTOFF / 2.0
                                )[0]:
                                    pointcoord = torch.tensor(
                                        [linx[x], liny[y], linz[z]]
                                    )
                                    occupancy[i, x, y, z] += atom_density(
                                        torch.linalg.norm(pointcoord - atomcoord),
                                        STD,
                                    )
        pname = structure_id + ".pickle"
        picklename = os.path.join(curr_path, pname)
        pickle_out = open(picklename, "wb")
        pickle.dump(occupancy, pickle_out)
        pickle_out.close()

print("Unprocessed files: ", (len(all_files) - proc_file))

end = time.time()


print("Time Elapsed:", end - start)
print(occupancy)
