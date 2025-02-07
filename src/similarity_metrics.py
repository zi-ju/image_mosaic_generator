import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# Calculate combined similarity metrics using SSIM and normalized MSE
def calculate_combined_similarity(mse_score, ssim_score, alpha=20, beta=0.5):
    # Normalize MSE (higher MSE means lower similarity)
    normalized_mse = max(0, 100 - alpha * np.log1p(mse_score))

    # Combine SSIM and normalized MSE
    similarity_percentage = beta * normalized_mse + (1 - beta) * max(0, ssim_score) * 100

    return similarity_percentage


# Calculate similarity metrics using SSIM
def calculate_similarity(original, mosaic):
    original_resized = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]))
    mse_score = ((original_resized - mosaic) ** 2).mean()
    ssim_score = ssim(original_resized, mosaic, channel_axis=2)

    # Calculate combined similarity
    similarity_percentage = calculate_combined_similarity(mse_score, ssim_score)

    # Provide qualitative interpretation
    if similarity_percentage > 80:
        interpretation = "Very similar 🎉"
    elif similarity_percentage > 60:
        interpretation = "Moderately similar 👍"
    elif similarity_percentage > 40:
        interpretation = "Noticeable differences 🧐"
    else:
        interpretation = "Low similarity 🙁"

    return mse_score, ssim_score, similarity_percentage, interpretation
