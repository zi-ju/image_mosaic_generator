import numpy as np


# # Generates the final mosaic image from the grid
def apply_mosaic(grid, cell_h, cell_w):
    grid_rows, grid_cols = grid.shape[:2]

    # Calculate the final mosaic size
    mosaic_h, mosaic_w = grid_rows * cell_h, grid_cols * cell_w
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    for i in range(grid_rows):
        for j in range(grid_cols):
            # Get the color of the current tile
            color = grid[i, j]
            # Fill the corresponding area in the mosaic
            mosaic[i * cell_h: (i + 1) * cell_h, j * cell_w: (j + 1) * cell_w] = color

    return mosaic
