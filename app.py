import gradio as gr
from PIL import Image
from src.image_processing import preprocess_image
from src.grid_processing import divide_into_grid
from src.mosaic_generator import apply_mosaic
from src.similarity_metrics import calculate_similarity


def generate_mosaic(input_image, grid_count):
    # Resize image and apply color quantization
    quantized_image = preprocess_image(input_image, max_dim=512, k_colors=8)

    # Divide image into grid
    grid, cell_h, cell_w = divide_into_grid(quantized_image, grid_count)

    # Generate mosaic
    mosaic = apply_mosaic(grid, cell_h, cell_w)

    # Calculate similarity metrics
    ssim_score, similarity_percentage, interpretation = calculate_similarity(input_image, mosaic)

    similarity_output = f"SSIM: {ssim_score:.2f} \nSimilarity: {similarity_percentage:.2f}% - {interpretation}"

    return Image.fromarray(mosaic), similarity_output


demo = gr.Interface(
    fn=generate_mosaic,
    inputs=[
        gr.Image(label="Original Image"),
        gr.Slider(2, 100, value=16, step=1, label="Grid number on width")],
    outputs=[
        gr.Image(label="Mosaic Image"),  # Mosaic image
        gr.Textbox(label="Similarity")  # Similarity metrics
    ],
    title="Interactive Image Mosaic Generator",
    description="Upload an image and generates a mosaic representation 🧩!",)

if __name__ == "__main__":
    demo.launch()
