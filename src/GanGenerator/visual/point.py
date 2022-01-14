import py3Dmol
import numpy as np


def draw_sphere(point_cloud, vobj=None, radius=1, rotate=0, color="red", display=True):
    if vobj is None:
        vobj = py3Dmol.view(width=800, height=600)

    if isinstance(point_cloud, np.ndarray):
        point_cloud = point_cloud.tolist()

    if all(isinstance(i, list) for i in point_cloud):
        for i in point_cloud:
            vobj.addSphere(
                {
                    "center": {
                        "x": i[0],
                        "y": i[1],
                        "z": i[2],
                    },
                    "radius": radius,
                    "color": f"{color}",
                }
            )
    else:
        vobj.addSphere(
            {
                "center": {
                    "x": point_cloud[0],
                    "y": point_cloud[1],
                    "z": point_cloud[2],
                },
                "radius": radius,
                "color": color,
            }
        )
    if display:
        vobj.show()
    return vobj


def draw_grid(vobj):

    return vobj


def draw_point(
    vobj,
    center,
    radius=1,
    rcolor="red",
    lcolor="blue",
    line=True,
    dash=False,
    display=True,
    line_pad=5,
    **kwargs,
):
    """Draw aline pass through the the centre of mass of protein and ligands

    Args:
        vobj (py3dmol): py3Dmol object
        center ([type]): [description]
        radius (int, optional): [description]. Defaults to 1.
        rcolor (str, optional): [description]. Defaults to "red".
        lcolor (str, optional): [description]. Defaults to "blue".
        line (bool, optional): [description]. Defaults to True.
        dash (bool, optional): [description]. Defaults to False.
        display (bool, optional): [description]. Defaults to True.
        line_pad (int, optional): [description]. Defaults to 5.

    Returns:
        [type]: [description]
    """
    return_coordinates = kwargs.get("return_coordinates", False)
    if isinstance(center, list):
        start, end = center[0], center[1]
    else:
        return "Center must be a list of xyz"
    vobj.addSphere(
        {
            "center": {"x": start[0], "y": start[1], "z": start[2]},
            "radius": radius,
            "color": f"{rcolor if rcolor is not None else 'red'}",
        }
    )
    vobj.addSphere(
        {
            "center": {"x": end[0], "y": end[1], "z": end[2]},
            "radius": radius,
            "color": f"{lcolor if lcolor is not None else 'blue'}",
        }
    )

    if line:
        # extends line with line_pad
        new_start = [(y - x) * line_pad + x for x, y in zip(start, end)]
        new_end = [(y - x) * line_pad + x for x, y in zip(end, start)]

        vobj.addLine(
            {
                "dashed": dash,
                "start": {
                    "x": new_start[0],
                    "y": new_start[1],
                    "z": new_start[2],
                },
                "end": {"x": new_end[0], "y": new_end[1], "z": new_end[2]},
            }
        )
        # vobj.addArrow(
        #    {
        #        "start": {
        #            "x": new_start[0],
        #            "y": new_start[1],
        #            "z": new_start[2],
        #        },
        #        "end": {"x": new_end[0], "y": new_end[1], "z": new_end[2]},
        #        "radius": 0.5,
        #    }
        # )

    if display:
        vobj.show()
    if return_coordinates:
        return new_start, new_end, vobj
    return vobj


def fill_sphere(vobj, mol):
    """
    Fill the pocket with spheres"""

    for i, atom in enumerate(mol):
        vobj.addSphere(
            {
                "center": {
                    "x": atom.coords[0],
                    "y": atom.coords[1],
                    "z": atom.coords[2],
                },
                "radius": 0.25,
                "color": f"{mol.atoms[i].color}",
                "clickable": True,
                "wireframe": True,
            }
        )

    return vobj


def shere_contact(mol):
    """Check so that no sphere overlap"""
    return


def draw_voxel(point_cloud, vobj=None, rotate=0):
    if vobj is None:
        vobj = py3Dmol.view(width=800, height=600)

    if isinstance(point_cloud, np.ndarray):
        point_cloud = point_cloud.tolist()

    voxel_box_center, voxel_box_size = point_cloud[0], point_cloud[1]
    try:
        vobj.addBox(
            {
                "center": {
                    "x": voxel_box_center[0],
                    "y": voxel_box_center[1],
                    "z": voxel_box_center[2],
                },
                "dimensions": {
                    "w": voxel_box_size[0],
                    "h": voxel_box_size[1],
                    "d": voxel_box_size[2],
                },
                "color": "grey",
                "opacity": 0.4,
            }
        )
        if rotate != 0:
            vobj.rotate(
                rotate,
                {
                    "x": voxel_box_center[0],
                    "y": voxel_box_center[1],
                    "z": voxel_box_center[2],
                },
            )
    except Exception as e:
        print("Failed to add Grid")

    return vobj
