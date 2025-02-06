import cv2
from skimage.metrics import structural_similarity as ssim


# Calculate similarity metrics using SSIM
def calculate_similarity(original, mosaic):
    original_resized = cv2.resize(original, (mosaic.shape[1], mosaic.shape[0]))
    ssim_score = ssim(original_resized, mosaic, channel_axis=2)

    # Convert SSIM score to similarity percentage
    similarity_percentage = max(0, ssim_score) * 100

    # Provide qualitative interpretation
    if similarity_percentage > 80:
        interpretation = "Very similar 🎉"
    elif similarity_percentage > 60:
        interpretation = "Moderately similar 👍"
    elif similarity_percentage > 40:
        interpretation = "Noticeable differences 🧐"
    else:
        interpretation = "Low similarity 🙁"

    return ssim_score, similarity_percentage, interpretation
