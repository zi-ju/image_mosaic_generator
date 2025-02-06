import cv2
import numpy as np
from sklearn.cluster import KMeans


# MAX_DIM = 512


# Resize and apply color quantization
def preprocess_image(image, resize_dim=(512, 512), k_colors=8):
    # Convert to OpenCV format and resize
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_bgr = cv2.resize(image_bgr, resize_dim, interpolation=cv2.INTER_AREA)

    # Reshape for K-Means
    pixels = image_bgr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k_colors, n_init=10, random_state=0)
    labels = kmeans.fit_predict(pixels)
    quantized_pixels = kmeans.cluster_centers_[labels].astype(np.uint8)

    # Reshape back
    quantized_image = quantized_pixels.reshape(image_bgr.shape)
    quantized_image_rgb = cv2.cvtColor(quantized_image, cv2.COLOR_BGR2RGB)
    return quantized_image_rgb
