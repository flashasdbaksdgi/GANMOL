import math
import numpy as np


def make_grid(tensor, nrow=8, padding=2, normalize=False, scale_each=False):

    nmaps = tensor.shape[0]
    ymaps = min(nrow, nmaps)
    xmaps = int(math.ceil(float(nmaps) / ymaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros(
        [height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3],
        dtype=np.uint8,
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h : h + h_width, w : w + w_width] = tensor[k]
            k = k + 1
    return grid
