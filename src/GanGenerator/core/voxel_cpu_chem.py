# Takes PDB file and outputs full maps for different atomtypes
# Uses GPU for voxelation
import os
import pickle
import plotly
import matplotlib.pyplot as plt
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import AllChem

from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np  # CPU
import rich
import seaborn as sns
from rich.console import Console
from rich.progress import track
from IPython import display

import torch
import time
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console = Console()

# import cupy as cp #FOR CUDA

# Settings:-------------------------------------------------------------------
#

ATOM_TYPES = {
    0: "Carbon",
    1: "Nitrogen",
    2: "Oxygen",
    3: "Sulphur",
    4: "Hydrogen",
    5: 'Phosphorus',
    6: 'Fluorine',
    7: 'Chlorine',
    8: 'Bromine',
    9: 'Iodine',
    10: 'Magnesium',
    11: 'Manganese',
    12: 'Zinc',
    13: 'Calcium',
    14: 'Iron',
    15: 'Boron',

}

# Defaults:
CUTOFF = 1.5  # CUTOFF (beyond which a point cannot feel an atom) (angstroms)
# the variance (typically 2.0 for a normal Gaussian distribution)
SIGMA = 1
V_SIZE = 0.5  # voxel size
PAD = 5  # PAD size around edges of protein in angstroms
# Number of types of atoms, must be aligned with atom_types.py
N_ATOMTYPES = len(ATOM_TYPES)
proc_file = 0  # number of files processed
DIM_SIZE = 60  # number of voxels in each dimension


def atom_id(atom: Chem.Atom) -> np.ndarray:

    id_mat = np.zeros([1, N_ATOMTYPES])[0]
    # Atom type 1: Carbon
    if atom.GetSymbol() == "C":
        id_mat[0] = 1

    # Atom type 2: Nitrogen
    if atom.GetSymbol() == "N":
        id_mat[1] = 1

    # Atom type 3: Oxygen
    if atom.GetSymbol() == "O":
        id_mat[2] = 1

    # Atom type 4: Sulfur
    if atom.GetSymbol() == "S":
        id_mat[3] = 1

    if atom.GetSymbol() == "H":
        id_mat[4] = 1

    if atom.GetSymbol() == "P":
        id_mat[5] = 1

    if atom.GetSymbol() == "F":
        id_mat[6] = 1

    if atom.GetSymbol() == "Cl":
        id_mat[7] = 1

    if atom.GetSymbol() == "Br":
        id_mat[8] = 1

    if atom.GetSymbol() == "I":
        id_mat[9] = 1

    if atom.GetSymbol() == "Mg":
        id_mat[10] = 1

    if atom.GetSymbol() == "Mn":
        id_mat[11] = 1

    if atom.GetSymbol() == "Zn":
        id_mat[12] = 1

    if atom.GetSymbol() == "Ca":
        id_mat[13] = 1

    if atom.GetSymbol() == "Fe":
        id_mat[14] = 1

    if atom.GetSymbol() == "B":
        id_mat[15] = 1

    return id_mat


# Calculate density felt at a point from a distance from an atom center:
def atom_density(distance: float, SIGMA: float) -> float:
    return np.exp(-(distance ** 2) / (2 * SIGMA ** 2))


def inverse_density(density: float, sigma: float) -> float:
    return -((2 * sigma**2)*np.log(density))**1/2


def atom_density1(r, sigma):
    return np.exp(-r*r/sigma/sigma/2)


def sigmoid_smoothing(array: np.ndarray) -> np.ndarray:
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            for k in range(np.shape(array)[2]):
                array[i, j, k] = 1 / (1 + np.exp(-array[i, j, k]))
    return array


def reverse_atom_density(array: np.ndarray, norm: float) -> np.ndarray:
    for i in array.shape[0]:
        for j in array.shape[1]:
            for k in array.shape[2]:
                array[i, j, k] = array[i, j, k] / norm

    return array

# get atom radius of an atom


def compute_centroid(coordinates) -> np.ndarray:
    """Compute the x,y,z centroid of provided coordinates
    coordinates: np.ndarray
      Shape (N, 3), where N is number atoms.
    """
    return np.mean(coordinates, axis=0, dtype=np.float32)


def atom_radii(atom: Chem.rdchem.Atom) -> float:
    """van der Waals radii:
    """
    return Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())


# Calculate center of geometry for a given residue:
def res_cog(residue):
    coord = [
        residue.get_list()[i].get_coord()
        for i in range(np.shape(residue.get_list())[0])
    ]

    return np.mean(coord, axis=0)


def atom_coordinate(atom, mol) -> list:
    atom_coords = mol.GetConformer().GetAtomPosition(atom.GetIdx())
    return [atom_coords.x, atom_coords.y, atom_coords.z]


def grid_construct(xyz, voxel_size, grid_center=None,
                   dimension=23.5, pad=2.0, pad_on=False):

    if grid_center is None:
        grid_center = compute_centroid(xyz)

    if grid_center.shape != (3,):
        raise ValueError("gridcenter must be a 3D coordinate")
    #    xmin, xmax = np.min(xyz[:, 0]), np.max(xyz[:, 0])
    #    ymin, ymax = np.min(xyz[:, 1]), np.max(xyz[:, 1])
    #    zmin, zmax = np.min(xyz[:, 2]), np.max(xyz[:, 2])
#
    # else:
    xmin = grid_center[0] - dimension/2
    xmax = grid_center[0] + dimension/2
    ymin = grid_center[1] - dimension/2
    ymax = grid_center[1] + dimension/2
    zmin = grid_center[2] - dimension/2
    zmax = grid_center[2] + dimension/2
    if pad_on:
        xmax += pad
        ymax += pad
        zmax += pad
        xmin -= pad
        ymin -= pad
        zmin -= pad
    X = np.array([xmax-xmin, ymax-ymin, zmax-zmin])
    # arange not stable(if dtype note mentioned) for float so use np.linspace
    linx = np.arange(xmin, xmax, voxel_size, dtype=np.float32)
    liny = np.arange(ymin, ymax, voxel_size, dtype=np.float32)
    linz = np.arange(zmin, zmax, voxel_size, dtype=np.float32)

    print("==========================================================")
    print(grid_center)
    # In the 3-D case with inputs of length M, N and P,
    # outputs are of shape(N, M, P) for ‘xy’ indexing and (M, N, P) for ‘ij’ indexing.
    gridx, gridy, gridz = np.meshgrid(linx, liny, linz, indexing="ij")

    return gridx, gridy, gridz, linx, liny, linz


def molecule_2coordinates(mol):
    coordinates = []
    for atom in range(mol.GetNumAtoms()):
        atom_coords = mol.GetConformer().GetAtomPosition(atom)
        coordinates.append(
            np.array([atom_coords.x, atom_coords.y,
                     atom_coords.z], dtype=np.float32)
        )
    return np.array(coordinates, dtype=np.float32)


def coordinates_2grid(coord_list, V_SIZE, PAD=5.):
    xmin = min(coord_list[i][0] for i in range(0, _shape))
    xmin = xmin - PAD
    xmax = max(coord_list[i][0] for i in range(_shape))
    xmax = xmax + PAD

    ymin = min([coord_list[i][1] for i in range(_shape)])
    ymin = ymin - PAD
    ymax = max(coord_list[i][1] for i in range(_shape))
    ymax = ymax + PAD

    zmin = min(coord_list[i][2] for i in range(0, _shape))
    zmin = zmin - PAD
    zmax = max(coord_list[i][2] for i in range(_shape))
    zmax = zmax + PAD

    linx = np.arange(xmin, xmax, V_SIZE)
    liny = np.arange(ymin, ymax, V_SIZE)
    linz = np.arange(zmin, zmax, V_SIZE)

    gridx, gridy, gridz = np.meshgrid(linx, liny, linz, sparse=True)
    return gridx, gridy, gridz, linx, liny, linz


# Load in PDB files:-----------------------------------------------------------


def runner(all_files, curr_path, DIM):

    for proc_file, item in enumerate(all_files):
        file, extension = os.path.splitext(item)
        if extension == ".pdb" and file == "2ogm":

            #print("Processing File", proc_file, file)
            filename = os.path.join(curr_path, item)
            molecule = Chem.MolFromPDBFile(filename, removeHs=False)

            # Populate a grid with atomic densities:--------------------------------------
            # Define grid edges
            coord_list = molecule_2coordinates(molecule)
            _shape = np.shape(coord_list)[0]
            gridx, gridy, gridz, linx, liny, linz = grid_construct(
                coord_list, V_SIZE, pad=PAD, dimension=DIM)
            gridshape = np.shape(gridx)

            occupancy = np.zeros(
                [N_ATOMTYPES, np.shape(linx)[0], np.shape(liny)[
                    0], np.shape(linz)[0]]
            )
            atom_count = 0

            for atom in track(molecule.GetAtoms(), description="Progress:"):
                atom_count += 1
                id_mat = atom_id(atom)
                # print(f"id_mat: {id_mat}\n atom: {atom}")
                # print(f"Atom radius: {atom_radii(atom)}")
                for i in range(N_ATOMTYPES):
                    if id_mat[i] == 1:
                        atomcoord = atom_coordinate(atom, molecule)
                        # Get Van der Waals radii (angstrom)
                        # TODO: check for bond and get radius
                        ATOM_RADIUS = atom_radii(atom)
                        for x in np.where(abs(linx - atomcoord[0]) < ATOM_RADIUS)[0]:
                            for y in np.where(abs(liny - atomcoord[1]) < ATOM_RADIUS)[0]:
                                for z in np.where(abs(linz - atomcoord[2]) < ATOM_RADIUS)[0]:
                                    pointcoord = np.array(
                                        [linx[x], liny[y], linz[z]])
                                    # print(pointcoord)
                                    distance = np.linalg.norm(
                                        pointcoord - atomcoord)
                                    original_point = atom_density(
                                        distance, SIGMA)
                                    occupancy[i, z, x, y
                                              ] += original_point
                                    # grid_losses_point = inverse_density(
                                    #    original_point, SIGMA)
                                    #print(distance, grid_losses_point)

    return occupancy, proc_file, atom_count


def grid_2coordinates(grid):
    coordinates = []
    print(np.shape(grid))
    # grid = np.linalg.norm(grid)
    print("grid coordinates")
    for i in range(grid.shape[0]):
        for x in range(np.shape(grid)[1]):
            for y in range(np.shape(grid)[2]):
                for z in range(np.shape(grid)[3]):

                    coordinates.append(
                        np.array(grid[i, x, y, z] *
                                 np.linalg.norm(grid[i, x, y, z]))
                    )
    return np.array(coordinates)


def show_animation_plot(occupancy, ATOM_TYPES):
    fig = px.imshow(occupancy, facet_col=0, animation_frame=1, facet_col_wrap=8,
                    binary_string=False, binary_format='jpg', height=800,
                    title="Slice Channel")
    for i in range(occupancy.shape[0]):
        fig.layout.annotations[i].text = f"{ATOM_TYPES[i]}"
    plotly.io.show(fig)


def show_plot(occupancy):
    fig = plt.figure(figsize=(30., 50.))
    imagegrid = ImageGrid(fig, 111,
                          nrows_ncols=(5, 5),
                          axes_pad=0.1,
                          )

    for ax, im in zip(imagegrid, occupancy[0, :, :, :]):
        ax.imshow(im)
    plt.show()


def tensor_2pdb(tensor, filename="tensor", format='pdb'):

    coordinates = getCoordinates(tensor)

    if format == 'pdb':
        Chem.MolToPDBFil(mol, filename)
    elif format == 'xyz':
        Chem.MolToXYZFile(mol, filename)
    else:
        Chem.MolToMolFile(mol, filename)
    return None


def main():

    # Identify file locations:
    curr_path = "./notebooks/grid"  # os.getcwd()

    pdb_path = os.path.join(curr_path, "chainA-clean.pdb")
    pickle_path = os.path.join(curr_path, "3pickle_perpdb")
    all_files = os.listdir(curr_path)
    start = time.time()
    occupancy, proc_file, atom_count = runner(all_files, curr_path, DIM_SIZE)
    print("Unprocessed files: ", (len(all_files) - proc_file))
    # print(occupancy)
    print(occupancy.shape)

    # ploting_plot
    end = time.time()
    console.print(
        f"[bold yellow]Total time of calculation [/bold yellow] [bold red]{atom_count}[bold red][bold yellow] atoms on a 4D grid[/bold yellow]: {end - start} [bold green]seconds[/bold green]")
    #show_animation_plot(occupancy, ATOM_TYPES)
    show_plot(occupancy)

    # with open(os.path.join(curr_path, "occupancy.pkl"), 'wb') as f:
    #    pickle.dump(occupancy, f)


def remain():
    curr_path = "./notebooks/grid"  # os.getcwd()

    pdb_path = os.path.join(curr_path, "chainA-clean.pdb")
    all_files = os.listdir(curr_path)
    with open(os.path.join(curr_path, "occupancy.pkl"), 'rb') as f:
        occupancy = pickle.load(f)

    print(occupancy.shape)
    # show_plot(occupancy)
    coordinates = grid_2coordinates(occupancy)
    print(coordinates.shape)
    print(coordinates[:, :, :, 0])

    # print(ATOM_TYPES)


if __name__ == "__main__":
    main()
    # remain()
