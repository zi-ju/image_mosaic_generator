import numpy as np


# Generates the final mosaic image from the grid
def apply_mosaic(grid, tile_size=32):
    grid_size = grid.shape[0]
    mosaic_h, mosaic_w = grid_size * tile_size, grid_size * tile_size
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    for i in range(grid_size):
        for j in range(grid_size):
            color = grid[i, j]
            mosaic[i * tile_size: (i + 1) * tile_size,
                   j * tile_size: (j + 1) * tile_size] = color

    return mosaic
