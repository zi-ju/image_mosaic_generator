import numpy as np


# Divide the image into grid and extracts average colors
def divide_into_grid(image, grid_size):
    h, w, _ = image.shape

    # Compute the best cell size that evenly divides the image
    cell_h = h // grid_size
    cell_w = w // grid_size

    # Recalculate grid size to fit the image exactly
    grid_rows = h // cell_h
    grid_cols = w // cell_w

    grid = []
    for i in range(grid_rows):
        row = []
        for j in range(grid_cols):
            cell = image[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            # Computes the average color of the cell
            avg_color = np.mean(cell, axis=(0, 1)).astype(np.uint8)
            row.append(avg_color)
        grid.append(row)

    return np.array(grid), cell_h, cell_w
