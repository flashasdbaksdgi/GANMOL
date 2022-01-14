import vapory
from PIL import Image
import math
import torch
from IPython.display import display, clear_output


def create_scene(moment):
    angle = 2 * math.pi * moment / 360
    r_camera = 7
    camera = vapory.Camera(
        "location",
        [r_camera * math.cos(angle), 1.5, r_camera * math.sin(angle)],
        "look_at",
        [0, 0, 0],
        "angle",
        30,
    )
    light1 = vapory.LightSource([2, 4, -3], "color", [1, 1, 1])
    light2 = vapory.LightSource([2, 4, 3], "color", [1, 1, 1])
    plane = vapory.Plane([0, 1, 0], -1, vapory.Pigment("color", [1, 1, 1]))
    box = vapory.Box(
        [0, 0, 0],
        [1, 1, 1],
        vapory.Pigment("Col_Glass_Clear"),
        vapory.Finish("F_Glass9"),
        vapory.Interior("I_Glass1"),
    )
    spheres = [
        vapory.Sphere(
            [float(r[0]), float(r[1]), float(r[2])],
            0.05,
            vapory.Texture(vapory.Pigment("color", [1, 1, 0])),
        )
        for r in x
    ]
    return vapory.Scene(
        camera, objects=[light1, light2, plane, box] + spheres, included=["glass.inc"]
    )


for t in range(0, 360):
    flnm = "out/sphere_{:03}.png".format(t)
    scene = create_scene(t)
    scene.render(flnm, width=800, height=600, remove_temp=False)
    clear_output()
    display(Image.open(flnm))

# convert -delay 10 -loop 0 sphere_*.png sphere_all.gif

# pytorch to make a sphere
def make_sphere(radius, center, color):
    sphere = torch.zeros(3, dtype=torch.float32)
    sphere[0] = radius
    sphere[1] = center[0]
    sphere[2] = center[1]
    sphere = torch.cat([sphere, color])
    return sphere
