import numpy as np


# Divides the image into grid and extracts average colors
def divide_into_grid(image, grid_size):
    h, w, _ = image.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    grid = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            cell = image[i * cell_h: (i + 1) * cell_h,
                         j * cell_w: (j + 1) * cell_w]
            # Computes the average color of the cell
            avg_color = np.mean(cell, axis=(0, 1)).astype(np.uint8)
            row.append(avg_color)
        grid.append(row)

    return np.array(grid)
