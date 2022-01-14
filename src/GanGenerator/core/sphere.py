from math import pi, sin, cos
from OpenGL.GL import *
from OpenGL.GLU import *


def sphere(lats, longs, r):
    for i in range(lats):
        lat0 = pi * (-0.5 + i / lats)
        z0 = sin(lat0)
        zr0 = cos(lat0)

        lat1 = pi * (-0.5 + (i + 1) / lats)
        z1 = sin(lat1)
        zr1 = cos(lat1)

        glBegin(GL_QUAD_STRIP)
        for j in range(longs + 1):
            lng = 2 * pi * (j + 1) / longs
            x = cos(lng)
            y = sin(lng)

            glNormal(x * zr0, y * zr0, z0)
            glVertex(r * x * zr0, r * y * zr0, r * z0)
            glNormal(x * zr1, y * zr1, z1)
            glVertex(r * x * zr1, r * y * zr1, r * z1)
        glEnd()


def main():
    lats = 10
    longs = 10
    r = 10
    sphere(lats, longs, r)


if __name__ == "__main__":
    main()
