import cv2
import numpy as np

class ImageResizer:
    def __init__(self, width=1935, height=1024):
        self.width = width
        self.height = height

    def __call__(self, image):
        return cv2.resize(image, (self.width, self.height))

class CLAHEEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(16, 16)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image):
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply CLAHE
        enhanced = self.clahe.apply(gray)

        # Convert back to RGB if input was RGB
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        return enhanced

def preprocess_image(image):
    """
    Preprocess the image using resize and CLAHE with exact specifications
    Args:
        image: Input image in BGR format
    Returns:
        Preprocessed image ready for model inference
    """
    # Create preprocessing pipeline
    resizer = ImageResizer(width=1935, height=1024)
    clahe_enhancer = CLAHEEnhancer(clip_limit=2.0, tile_grid_size=(16, 16))

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply pipeline steps
    img_resized = resizer(img_rgb)
    img_enhanced = clahe_enhancer(img_resized)

    return img_enhanced