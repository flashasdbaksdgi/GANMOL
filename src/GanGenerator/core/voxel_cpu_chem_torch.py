# Takes PDB file and outputs full maps for different atomtypes
# Uses GPU for voxelation
import os
import pickle
import plotly
import torch
import time
import matplotlib.pyplot as plt
import plotly.express as px
from rdkit import Chem
from mpl_toolkits.axes_grid1 import ImageGrid
from rich.console import Console
from rich.progress import track


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
# CUTOFF = 1.5  # CUTOFF (beyond which a point cannot feel an atom) (angstroms)
# the variance (typically 2.0 for a normal Gaussian distribution)
SIGMA = 1
V_SIZE = 0.5  # voxel size
PAD = 5  # PAD size around edges of protein in angstroms
# Number of types of atoms, must be aligned with atom_types.py
N_ATOMTYPES = len(ATOM_TYPES)
DIM_SIZE = 200  # number of voxels in each dimension


def atom_id(atom: Chem.Atom) -> torch.tensor:

    id_mat = torch.zeros([1, N_ATOMTYPES], dtype=torch.int32)[0]
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
    return torch.exp(-(distance ** 2) / (2 * SIGMA ** 2))


def inverse_density(density: float, sigma: float) -> float:
    return -((2 * sigma**2)*torch.log(density))**1/2


def atom_density1(r, sigma):
    return torch.exp(-r*r/sigma/sigma/2)


# get atom radius of an atom

def compute_centroid(coordinates) -> torch.tensor:
    """Compute the x,y,z centroid of provided coordinates
    coordinates: torch.ndarray
      Shape (N, 3), where N is number atoms.
    """
    return torch.mean(coordinates, axis=0, dtype=torch.float32)


def atom_radii(atom: Chem.rdchem.Atom) -> float:
    """van der Waals radii:
    """
    return Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())


# Calculate center of geometry for a given residue:
def res_cog(residue):
    coord = [
        residue.get_list()[i].get_coord()
        for i in range(torch.shape(residue.get_list())[0])
    ]

    return torch.mean(coord, axis=0)


def atom_coordinate(atom, mol) -> list:
    atom_coords = mol.GetConformer().GetAtomPosition(atom.GetIdx())
    return torch.tensor([atom_coords.x, atom_coords.y, atom_coords.z])


def grid_construct(xyz, voxel_size, grid_center=None,
                   dimension=23.5, pad=2.0, pad_on=False, device=None):

    if grid_center is None:
        grid_center = compute_centroid(xyz)

    if grid_center.shape != (3,):
        raise ValueError("gridcenter must be a 3D coordinate")
    #    xmin, xmax = torch.min(xyz[:, 0]), torch.max(xyz[:, 0])
    #    ymin, ymax = torch.min(xyz[:, 1]), torch.max(xyz[:, 1])
    #    zmin, zmax = torch.min(xyz[:, 2]), torch.max(xyz[:, 2])
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
    X = torch.tensor([xmax-xmin, ymax-ymin, zmax-zmin])
    # arange not stable(if dtype note mentioned) for float so use torch.linspace
    linx = torch.arange(xmin, xmax, voxel_size,
                        device=device, dtype=torch.float32)
    liny = torch.arange(ymin, ymax, voxel_size,
                        device=device, dtype=torch.float32)
    linz = torch.arange(zmin, zmax, voxel_size,
                        device=device, dtype=torch.float32)

    # print("==========================================================")
    # print(grid_center)
    # In the 3-D case with inputs of length M, N and P,
    # outputs are of shape(N, M, P) for ‘xy’ indexing and (M, N, P) for ‘ij’ indexing.
    gridx, gridy, gridz = torch.meshgrid(linx, liny, linz, indexing="ij")

    return gridx, gridy, gridz, linx, liny, linz


def molecule_2coordinates(mol):
    coordinates = []
    for atom in range(mol.GetNumAtoms()):
        atom_coords = mol.GetConformer().GetAtomPosition(atom)
        coordinates.append(
            ([atom_coords.x, atom_coords.y,
              atom_coords.z])
        )

    return torch.tensor(coordinates, dtype=torch.float32)


# Load in PDB files:-----------------------------------------------------------


def runner(all_files, curr_path, DIM, device=None):

    for proc_file, item in enumerate(all_files):
        file, extension = os.path.splitext(item)
        if extension == ".pdb" and file == "2ogm":

            # print("Processing File", proc_file, file)
            filename = os.path.join(curr_path, item)
            molecule = Chem.MolFromPDBFile(filename, removeHs=False)

            # Populate a grid with atomic densities:--------------------------------------
            # Define grid edges
            coord_list = molecule_2coordinates(molecule)
            _shape = coord_list.size()[0]
            gridx, gridy, gridz, linx, liny, linz = grid_construct(
                coord_list, V_SIZE, pad=PAD, dimension=DIM, device=device)
            gridshape = gridx.size

            occupancy = torch.zeros(
                [N_ATOMTYPES, linx.size()[0], liny.size()[
                    0], linz.size()[0]], dtype=torch.float32, device=device)

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
                        # print(type(atomcoord), 1)
                        # print(type(linx), 2)
                        for x in torch.where(abs(linx - atomcoord[0]) < ATOM_RADIUS)[0]:
                            for y in torch.where(abs(liny - atomcoord[1]) < ATOM_RADIUS)[0]:
                                for z in torch.where(abs(linz - atomcoord[2]) < ATOM_RADIUS)[0]:
                                    pointcoord = torch.tensor(
                                        [linx[x], liny[y], linz[z]])
                                    # print(type(pointcoord))
                                    distance = torch.linalg.norm(
                                        pointcoord - atomcoord)
                                    original_point = atom_density(
                                        distance, SIGMA)
                                    occupancy[i, x, y, z] += original_point
                                    # grid_losses_point = inverse_density(
                                    #    original_point, SIGMA)
                                    # print(distance, grid_losses_point)

    return occupancy, proc_file, atom_count


def grid_2coordinates(grid):
    coordinates = []
    print(torch.shape(grid))
    # grid = torch.linalg.norm(grid)
    print("grid coordinates")
    for i in range(grid.shape[0]):
        for x in range(torch.shape(grid)[1]):
            for y in range(torch.shape(grid)[2]):
                for z in range(torch.shape(grid)[3]):

                    coordinates.append(
                        torch.tensor(grid[i, x, y, z] *
                                     torch.linalg.norm(grid[i, x, y, z]))
                    )
    return torch.tensor(coordinates)


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


def forward_plot(occupancy):

    return None


def loss_function(occupancy):

    return None


def main():

    # Identify file locations:
    curr_path = "./notebooks/grid"  # os.getcwd()

    #pdb_path = os.path.join(curr_path, "chainA-clean.pdb")
    #pickle_path = os.path.join(curr_path, "3pickle_perpdb")
    all_files = os.listdir(curr_path)
    start = time.time()
    occupancy, proc_file, atom_count = runner(
        all_files, curr_path, DIM_SIZE, device=device)
    print("Unprocessed files: ", (len(all_files) - proc_file))
    # print(occupancy)
    print(occupancy.shape)

    # ploting_plot
    end = time.time()
    console.print(
        f"[bold yellow]Total time of calculation [/bold yellow] [bold red]{atom_count}[bold red][bold yellow] atoms on a 4D grid[/bold yellow]: {end - start} [bold green]seconds[/bold green]")
    occupancy = occupancy.detach().cpu().numpy()
    show_animation_plot(occupancy, ATOM_TYPES)
    # show_plot(occupancy)

    # with open(os.path.join(curr_path, "occupancy.pkl"), 'wb') as f:
    #    pickle.dump(occupancy, f)


def remain():
    curr_path = "./notebooks/grid"  # os.getcwd()

    pdb_path = os.path.join(curr_path, "chainA-clean.pdb")
    all_files = os.listdir(curr_path)
    with open(os.path.join(curr_path, "occupancy.pkl"), 'rb') as f:
        occupancy = pickle.load(f)

    print(occupancy.shape)
    show_plot(occupancy)
    coordinates = grid_2coordinates(occupancy)
    print(coordinates.shape)
    print(coordinates[:, :, :, 0])

    # print(ATOM_TYPES)


if __name__ == "__main__":
    main()
    # remain()
