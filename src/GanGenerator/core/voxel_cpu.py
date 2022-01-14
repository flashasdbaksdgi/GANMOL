# Takes PDB file and outputs full maps for different atomtypes
# Uses GPU for voxelation

import plotly.express as px
import plotly
from matplotlib import colors
from mpl_toolkits.axes_grid1 import ImageGrid
import os

import numpy as np  # CPU

# import cupy as cp #FOR CUDA
import rdkit
from rdkit import Chem
import pickle
from Bio.PDB.PDBParser import PDBParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rich.console import Console
from rich.progress import track
console = Console()

parser = PDBParser(PERMISSIVE=1)

# Settings:-------------------------------------------------------------------
#
# Defaults:
CUTOFF = 1.5  # CUTOFF (beyond which a point cannot feel an atom) (angstroms)
STD = 2.0  # the variance (typically 2.0 for a normal Gaussian distribution)
V_SIZE = 0.5  # voxel size
PAD = 10  # PAD size around edges of protein in angstroms
N_ATOMTYPES = 4  # Number of types of atoms, must be aligned with atom_types.py
proc_file = 0  # number of files processed

# Identify file locations:
curr_path = "./notebooks/grid"  # os.getcwd()

pdb_path = os.path.join(curr_path, "complete_complex.pdb")
pickle_path = os.path.join(curr_path, "3pickle_perpdb")
all_files = os.listdir(curr_path)


def atom_id(atom: Chem.Atom) -> np.ndarray:
    N_ATOMTYPES = 4
    id_mat = np.zeros([1, N_ATOMTYPES])[0]
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


ATOM_TYPES = {
    0: "Carbon",
    1: "Nitrogen",
    2: "Oxygen",
    3: "Sulphur",

}


# Calculate density felt at a point from a distance from an atom center:
def atom_density(distance: float, STD: float) -> float:
    density = np.exp(-(distance ** 2) / (2 * STD ** 2))
    # print("density")
    # print(density)
    return density


def reverse_atom_density(array, norm):
    for i in array.shape[0]:
        for j in array.shape[1]:
            for k in array.shape[2]:
                array[i, j, k] = array[i, j, k] / norm

    return array

# get atom radius of an atom


def propogation_center(coordinate1: list, coordinate2: list) -> list:
    # center of grid, to propograte from
    cx = (coorindate1[0] + coordinate2[0])/2
    cy = (coorindate1[1] + coordinate2[1])/2
    cz = (coorindate1[2] + coordinate2[2])/2
    return list(cx, cy, cz)


def atom_radii(mol: Chem.rdchem.Mol) -> list:
    radii = []
    for atom in mol.GetAtoms():
        atom_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
        radii.append(atom_radius)
    return radii


# Calculate center of geometry for a given residue:
def res_cog(residue):
    coord = [
        residue.get_list()[i].get_coord()
        for i in range(0, np.shape(residue.get_list())[0])
    ]
    cog = np.mean(coord, axis=0)
    return cog


# Load in PDB files:-----------------------------------------------------------

for item in all_files:
    file, extension = os.path.splitext(item)
    # and file[-6:] == "-clean":
    if extension == ".pdb" and file == "chainA":
        proc_file += 1

        print("Processing File", proc_file, file)

        structure_id = file
        filename = os.path.join(curr_path, item)
        structure = parser.get_structure(structure_id, filename)

        # Populate a grid with atomic densities:--------------------------------------
        # Define grid edges
        coord_list = [atom.coord for atom in structure.get_atoms()]
        # print(coord_list)
        # print(np.shape(coord_list)[0])
        # var = np.tensor(np.array(coord_list), dtype=np.float32)
        _shape = np.shape(coord_list)[0]
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

        linx = np.arange(xmin, xmax, V_SIZE)
        liny = np.arange(ymin, ymax, V_SIZE)
        linz = np.arange(zmin, zmax, V_SIZE)

        gridx, gridy, gridz = np.meshgrid(linx, liny, linz)
        gridshape = np.shape(gridx)
        # print("___sadsa")
        # print(gridx, gridy, gridz)
        # print("___sadsa", gridshape)
        # Fill densities into grid
        occupancy = np.zeros(
            [N_ATOMTYPES, np.shape(linx)[0], np.shape(liny)[
                0], np.shape(linz)[0]]
        )
#
        for residue in track(list(structure.get_residues()), description="Progress"):
            # print("Working on Residue", residue)
            for atom in residue.get_list():
                id_mat = atom_id(atom)
                # print(f"id_mat: {id_mat}\n atom: {atom}")
                # print(f"Atom radius: {atom_radii(atom)}")
                for i in range(0, N_ATOMTYPES):
                    if id_mat[i] == 1:
                        atomcoord = atom.get_coord()
                        for x in np.where(abs(linx - atomcoord[0]) < CUTOFF)[0]:
                            for y in np.where(abs(liny - atomcoord[1]) < CUTOFF)[0]:
                                for z in np.where(abs(linz - atomcoord[2]) < CUTOFF)[0]:
                                    pointcoord = np.array(
                                        [linx[x], liny[y], linz[z]])
                                    # print(pointcoord)
                                    normalized = np.linalg.norm(
                                        pointcoord - atomcoord)
                                    # console.print(
                                    #    normalized, style="blue underline")
                                    occupancy[i, x, y, z] += atom_density(
                                        normalized, STD)

        # saving to picklename = os.path.join(curr_path, pname)
        # pname = structure_id + ".pickle"
        # picklename = os.path.join(curr_path, pname)
        # pickle_out = open(picklename, "wb")
        # pickle.dump(occupancy, pickle_out)
        # pickle_out.close()

# densities groiod back to atom coordinates


def grid_2coordinates(grid):
    coordinates = []
    print(np.shape(grid))
    # grid = np.linalg.norm(grid)
    print("grid coordinates")
    print(grid)
    for i in range(0, np.shape(grid)[0]):
        for j in range(0, np.shape(grid)[1]):
            for k in range(0, np.shape(grid)[2]):
                console.print(grid[i, j, k])
                print("tetstiubg")
                coordinates.append(
                    np.array(grid[i, j, k] * np.linalg.norm(grid[i, j, k]))
                )
    return np.array(coordinates)


console.print(
    f"Unprocessed files: {(len(all_files) - proc_file)}", style="white on blue")
console.print(occupancy, style="black on white")
console.print(occupancy.shape)


# reverse
t1 = occupancy[0, :, :, :]
print(t1)
print(t1.shape)


def reverse_grid2coordinates(grid):
    x = grid[0, :, :]
    y = grid[1, :, :]
    z = grid[2, :, :]
    return np.concatenate(x, y, z, axis=-1)


# print(reverse_grid2coordinates(np.linalg.cond(t1)))
print("------------------------------------------------")
print((np.linalg.cond(occupancy[:, :, :, :])))
print("------------------------------------------------")

# ploting_plot

fig = px.imshow(occupancy, facet_col=0, animation_frame=1, facet_col_wrap=2,
                binary_string=False, binary_format='jpg', height=800,
                title="Slice Channel")
for i in range(occupancy.shape[0]):
    fig.layout.annotations[i].text = f"{ATOM_TYPES[i]}"
plotly.io.show(fig)
