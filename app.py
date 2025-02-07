import gradio as gr
from PIL import Image
from src.image_processing import preprocess_image
from src.grid_processing import divide_into_grid
from src.mosaic_generator import apply_mosaic_with_color, apply_mosaic_with_tiles, load_tile_images
from src.similarity_metrics import calculate_similarity


MAX_DIM = 5120
K_COLORS = 8
UNIQUENESS_THRESHOLD = 30


def generate_mosaic(input_image, grid_count, mosaic_type):
    # Resize image and apply color quantization
    quantized_image = preprocess_image(input_image, max_dim=MAX_DIM, k_colors=K_COLORS)

    # Divide image into grid
    grid, tile_h, tile_w = divide_into_grid(quantized_image, grid_count)

    if mosaic_type == "Color Blocks":
        # Generate mosaic with color blocks
        mosaic = apply_mosaic_with_color(grid, tile_h, tile_w)
    elif mosaic_type == "Image Tiles":
        # Generate mosaic with image tiles
        tile_images, tile_image_colors = load_tile_images("tiles/cats", tile_h, tile_w)
        mosaic = apply_mosaic_with_tiles(grid, tile_images, tile_image_colors, tile_h, tile_w, uniqueness_threshold=UNIQUENESS_THRESHOLD)

    # Calculate similarity metrics
    mse_score, ssim_score, similarity_percentage, interpretation = calculate_similarity(input_image, mosaic)

    similarity_output = f"MSE: {mse_score:.2f} \nSSIM: {ssim_score:.2f} \nSimilarity: {interpretation}"

    return Image.fromarray(mosaic), similarity_output


demo = gr.Interface(
    fn=generate_mosaic,
    inputs=[
        gr.Image(label="Original Image"),
        gr.Slider(2, 100, value=16, step=1, label="Grid number on width"),
        gr.Dropdown(["Color Blocks", "Image Tiles"], value="Color Blocks", label="Mosaic Type")
    ],
    outputs=[
        gr.Image(label="Mosaic Image"),  # Mosaic image
        gr.Textbox(label="Similarity")  # Similarity metrics
    ],
    title="Interactive Image Mosaic Generator",
    description="Upload an image and generates a mosaic representation 🧩!",)

if __name__ == "__main__":
    demo.launch()
