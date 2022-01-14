import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np

# interactive mode off, can [normally] be safely removed
plt.ioff()

# define an arrow class:
class Arrow3D(FancyArrowPatch):
    def __init__(self, start=[0, 0, 0], end=[1, 1, 1], *args, **kwargs):
        if "arrowstyle" not in kwargs:
            kwargs["arrowstyle"] = "-|>"
        if "mutation_scale" not in kwargs:
            kwargs["mutation_scale"] = 20
        if "color" not in kwargs:
            kwargs["color"] = "k"
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        zs = [start[2], end[2]]
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def WireframeSphere(
    centre=[0.0, 0.0, 0.0], radius=1.0, n_meridians=20, n_circles_latitude=None
):
    """
    Create the arrays of values to plot the wireframe of a sphere.

    Parameters
    ----------
    centre: array like
        A point, defined as an iterable of three numerical values.
    radius: number
        The radius of the sphere.
    n_meridians: int
        The number of meridians to display (circles that pass on both poles).
    n_circles_latitude: int
        The number of horizontal circles (akin to the Equator) to display.
        Notice this includes one for each pole, and defaults to 4 or half
        of the *n_meridians* if the latter is larger.

    Returns
    -------
    sphere_x, sphere_y, sphere_z: arrays
        The arrays with the coordinates of the points to make the wireframe.
        Their shape is (n_meridians, n_circles_latitude).

    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> sphere = ax.plot_wireframe(*WireframeSphere(), color="r", alpha=0.5)
    >>> fig.show()

    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> frame_xs, frame_ys, frame_zs = WireframeSphere()
    >>> sphere = ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.5)
    >>> fig.show()
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians / 2, 4)
    u, v = np.mgrid[
        0 : 2 * np.pi : n_meridians * 1j, 0 : np.pi : n_circles_latitude * 1j
    ]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z


def find_most_distants(points, center=[0.0, 0.0, 0.0], tol=1e-5):
    """
    Finds and returns a list of points that are the most distante ones to
    the center.

    Parameters
    ----------
    points: list
        A list of points (see center to know what a point is)
    center: array like
        A point, defined as an iterable of three numerical values.
    """
    # make central point an array to ease vector calculations
    center = np.asarray(center)
    # find most distant points
    max_distance = 0
    most_distant_points = []
    for point in points:
        distance = np.linalg.norm(center - point)
        if abs(distance - max_distance) <= tol:
            most_distant_points.append(point)
        elif distance > max_distance:
            most_distant_points = [point]
            max_distance = distance
    return max_distance, most_distant_points


def list_of_points_TO_lists_of_coordinates(list_of_points):
    """
    Converts a list of points to lists of coordinates of those points.

    Parameter
    ---------
    list_of_points: list
        A list of points (each defined as an iterable of three numerical values)

    Returns
    -------
    points_x, points_y, points_z: array
        Lists of coordinates
    """
    points_x = []
    points_y = []
    points_z = []
    for point in list_of_points:
        points_x.append(point[0])
        points_y.append(point[1])
        points_z.append(point[2])
    return points_x, points_y, points_z


def function(
    central_point=[0.0, 0.0, 0.0],
    other_points=[[1.0, 2.0, 2.23], [2.0, 3.0, 3.6], [-3.0, 4.0, 5.0]],
):
    """
    Draws a wireframe sphere centered on central_point and containing all
    points in other_points list. Also draws the points inside the sphere and
    marks the most distant ones with an arrow.

    Parameters
    ----------
    central_point: array like
        A point, defined as an iterable of three numerical values.
    other_points: list
        A list of points (see central_point to know what a point is)
    """
    # find most distant points
    max_distance, most_distant_points = find_most_distants(other_points, central_point)
    # prepare figure and 3d axis
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_aspect("equal")
    # draw sphere
    ax.plot_wireframe(
        *WireframeSphere(central_point, max_distance), color="r", alpha=0.5
    )
    # draw points
    ax.scatter(*list_of_points_TO_lists_of_coordinates(other_points))
    # draw arrows to most distant points:
    for extreme_point in most_distant_points:
        ax.add_artist(Arrow3D(start=central_point, end=extreme_point))
    fig.show()


if __name__ == "__main__":
    function([0, 0, 0], 2 * np.random.rand(50, 3) - 1)
    # make a list with equally most distant point:
    repeated_max_list = 2 * np.random.rand(10, 3) - 1
    distance, points = find_most_distants(repeated_max_list)
    repeated_max_list = np.concatenate((repeated_max_list, points))
    repeated_max_list[-1][0] = -repeated_max_list[-1][0]
    repeated_max_list[-1][1] = -repeated_max_list[-1][1]
    repeated_max_list[-1][2] = -repeated_max_list[-1][2]
    function([0, 0, 0], repeated_max_list)
