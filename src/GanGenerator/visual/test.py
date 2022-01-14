# give  a a line of two points in 3D space and center of a cube.
# function to make the line perpendiculator to the center of the box and cube surface
import numpy as np


def cube_perpendicular(center, point1, point2):
    """
    Calculate the perpendicular line of two points in 3D space.
    """
    # Calculate the distance between two points
    point1 = np.array(point1)
    point2 = np.array(point2)
    d = np.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
        + (point1[2] - point2[2]) ** 2
    )
    # Calculate the unit vector of the distance
    unit_vector = (point1 - point2) / d
    # Calculate the perpendicular vector of the unit vector
    perpendicular_vector = np.array([unit_vector[1], -unit_vector[0], 0])
    # Calculate the perpendicular line
    perpendicular_line = point1 + perpendicular_vector * d
    # Calculate the center of the cube
    cube_center = (point1 + point2) / 2
    # Calculate the perpendicular line to the center of the cube
    cube_perpendicular_line = cube_center + perpendicular_vector * d
    # Calculate the distance between the perpendicular line and the center of the cube
    cube_distance = np.sqrt(
        (cube_perpendicular_line[0] - cube_center[0]) ** 2
        + (cube_perpendicular_line[1] - cube_center[1]) ** 2
        + (cube_perpendicular_line[2] - cube_center[2]) ** 2
    )
    # Calculate the distance between the perpendicular line and the center of the box
    box_distance = np.sqrt(
        (perpendicular_line[0] - center[0]) ** 2
        + (perpendicular_line[1] - center[1]) ** 2
        + (perpendicular_line[2] - center[2]) ** 2
    )
    # Calculate the distance between the perpendicular line and the center of the box
    if box_distance < cube_distance:
        return perpendicular_line
    else:
        return cube_perpendicular_line
