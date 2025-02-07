# Interactive Image Mosaic Generator

## Overview
This project is an interactive web application that allows users to generate a mosaic representation of an image using either color blocks or small image tiles. The application is built with Python, OpenCV, NumPy, and Gradio for the user interface.

## Features
- Upload an image and generate a mosaic representation.
- Choose between color blocks or image tiles for the mosaic.
- Adjust the grid size dynamically.
- Compute and display similarity metrics (SSIM) between the original and mosaic images.
- Optimized image resizing and cropping to maintain aspect ratio.

## Installation
### Prerequisites
Ensure you have Python installed (version 3.8 or later recommended). Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy`
- `opencv-python`
- `gradio`
- `PIL` (Pillow)
- `scikit-image`

## Usage
Run the application with:

```bash
python app.py
```
OR

```bash
gradio app.py
```

Once the application starts, a Gradio interface will launch in your browser.


## How It Works
1. **Image Preprocessing**: Resizes the uploaded image while maintaining the aspect ratio.
2. **Grid Division**: Splits the image into a grid based on the chosen grid size.
3. **Mosaic Generation**:
   - If "Color Blocks" is selected, each cell is filled with the average color.
   - If "Image Tiles" is selected, tiles are matched to the grid using color similarity.
4. **Similarity Metrics**: Structural Similarity Index (SSIM) is computed to measure how closely the mosaic resembles the original image.


## Future Improvements
- Improve tile selection to reduce repetition.
- Optimize performance for large images.
- Add more advanced similarity metrics.
- Support different mosaic styles.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
- OpenCV for image processing.
- Gradio for building the interactive UI.
- Image datasets for mosaic generation from Kaggle: https://www.kaggle.com/datasets/chetankv/dogs-cats-images
