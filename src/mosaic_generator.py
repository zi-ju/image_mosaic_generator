import numpy as np
import cv2
import os


# Generates the final mosaic image from the grid
# with color blocks
def apply_mosaic_with_color(grid, cell_h, cell_w):
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


# Resize image to fit target size by maintaining aspect ratio
def resize_keep_ratio(image, target_h, target_w):
    h, w, _ = image.shape
    scale = max(target_h / h, target_w / w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# Crop the center portion of the image and fit to the target size
def crop_center(image, target_h, target_w):
    h, w, _ = image.shape

    # Ensure the target dimensions are smaller than the original image size
    target_h = min(target_h, h)
    target_w = min(target_w, w)

    # Find the center coordinates
    center_x, center_y = w // 2, h // 2

    # Calculate the cropping coordinates
    x1 = center_x - target_w // 2
    x2 = center_x + target_w // 2
    y1 = center_y - target_h // 2
    y2 = center_y + target_h // 2

    # Crop the image
    cropped_tile = image[y1:y2, x1:x2]
    cropped_tile = cv2.resize(cropped_tile, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return cropped_tile


# Load tile images from folder and crop the center portion
# Compute the average color of each tile for matching
def load_tile_images(tile_folder, tile_h, tile_w):
    cropped_tiles = []
    tile_image_colors = []

    for filename in os.listdir(tile_folder):
        filepath = os.path.join(tile_folder, filename)
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape

        # Ensure the image is large enough to crop
        if h < tile_h or w < tile_w:
            continue

        # Resize the image to fit the tile size but maintain aspect ratio
        img_resized = resize_keep_ratio(img, tile_h, tile_w)

        # Crop the center portion of the image
        cropped_tile = crop_center(img_resized, tile_h, tile_w)

        if cropped_tile.shape[:2] == (tile_h, tile_w):
            cropped_tiles.append(cropped_tile)
            tile_image_colors.append(np.mean(cropped_tile, axis=(0, 1)))  # Compute average color

    return cropped_tiles, tile_image_colors


# Find the best matching tile for a given cell color
# while avoiding repetition
def find_best_matching_tile(grid_color, tile_image_colors, used_tiles, uniqueness_threshold):
    # Compute the distance between the grid color and each tile color
    if len(tile_image_colors) == 0:
        raise ValueError("No tile images were loaded. Check the tile image folder.")
    distances = np.linalg.norm(np.array(tile_image_colors) - grid_color, axis=1)
    # Sort by closest color match
    sorted_indices = np.argsort(distances)

    # Find the first unused tile
    for idx in sorted_indices:
        if idx not in used_tiles:
            best_match_idx = idx
            break
    # If all are used, pick the closest
    else:
        best_match_idx = sorted_indices[0]

    used_tiles.add(best_match_idx)
    if len(used_tiles) > uniqueness_threshold:
        used_tiles.pop()

    return best_match_idx


# Generates the final mosaic image from the grid
# with image tiles
def apply_mosaic_with_tiles(grid, tile_images, tile_image_colors, tile_h, tile_w, uniqueness_threshold):
    grid_h, grid_w = grid.shape[:2]
    mosaic_h, mosaic_w = grid_h * tile_h, grid_w * tile_w
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    # Tracks recently used tile indices
    used_tiles = set()

    for i in range(grid_h):
        for j in range(grid_w):
            grid_color = grid[i, j]
            best_match_idx = find_best_matching_tile(grid_color, tile_image_colors, used_tiles, uniqueness_threshold)
            best_tile = tile_images[best_match_idx]

            # Place the tile in the mosaic
            mosaic[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w] = best_tile

    return mosaic
