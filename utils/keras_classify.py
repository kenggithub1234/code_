import numpy as np
import cv2


def get_model_input_size(model):
    """Best-effort extraction of the (height, width) a Keras model expects,
    read from model.input_shape. Falls back to 224x224 if it can't be
    determined (some models report shapes like (None, None, None, 3) when
    built with a fully-convolutional / global-pooling head)."""
    try:
        shape = model.input_shape
        if isinstance(shape, list):  # multi-input models - use the first
            shape = shape[0]
        _, h, w, _ = shape  # (None, H, W, C)
        if h and w:
            return int(h), int(w)
    except Exception:
        pass
    return 224, 224


def preprocess_image(img_bgr, size, mode="rescale_0_1"):
    """img_bgr: OpenCV BGR image (any size), will be resized to `size`.
    size: (h, w).
    mode:
      - 'rescale_0_1'  : divide by 255 (common default)
      - 'rescale_-1_1' : x/127.5 - 1 (common for MobileNet-style models)
      - 'raw'          : no scaling - use this if the model already has its
        own internal Rescaling/Normalization layer
    Returns a (1, H, W, 3) float32 RGB batch ready for model.predict().
    """
    h, w = size
    resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32")
    if mode == "rescale_0_1":
        rgb = rgb / 255.0
    elif mode == "rescale_-1_1":
        rgb = rgb / 127.5 - 1.0
    # 'raw' -> leave pixel values as-is
    return np.expand_dims(rgb, axis=0)
