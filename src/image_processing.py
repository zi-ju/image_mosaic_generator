import cv2
import numpy as np
from sklearn.cluster import KMeans


# Resize if needed and apply color quantization
def preprocess_image(image, max_dim, k_colors=8):
    # Convert to OpenCV format
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get original dimensions
    h, w = image_bgr.shape[:2]

    # Check if resizing is needed
    if max(h, w) > max_dim:
        # Resize while maintaining aspect ratio
        scale_factor = max_dim / max(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Reshape for K-Means
    pixels = image_bgr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k_colors, n_init=10, random_state=0)
    labels = kmeans.fit_predict(pixels)
    quantized_pixels = kmeans.cluster_centers_[labels].astype(np.uint8)    

    # Reshape back
    quantized_image = quantized_pixels.reshape(image_bgr.shape)
    quantized_image_rgb = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB)
    return quantized_image_rgb
