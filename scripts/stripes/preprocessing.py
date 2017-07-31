import numpy as np


# Size of details (in mm)
STRIPE_A = 5.
STRIPE_B = 120.

GAP_WIDTH = 2.45
POOL_WIDTH = 5.0

LEFT_MARGIN = 29.7
SIDE_MARGIN = 3.


def get_reference_rect(image):
    """
    Take an image (RGB or grayscale, whatever) of stripped test stripe and return 'white' reference rectangle. 
    """
    a, b = image.shape[:2]
    k = a / STRIPE_A
    return np.array([SIDE_MARGIN, SIDE_MARGIN,
                     (LEFT_MARGIN - SIDE_MARGIN) * k, a - SIDE_MARGIN * k], dtype=np.uint8)


def adjust_white_balance_wrt_rgb(image, (r_w, g_w, b_w), white_value=255):
    """
    Take an RGB-image with (r_w, g_w, b_w) being a RGB values of a 'truly white' object on this image,
    then scale all channels in a way that these values become white_value.
    """

    assert len(image.shape) == 3, "Not a 3-channel image"

    image_wb = image.copy()

    r_coeff = 1. * white_value / r_w
    g_coeff = 1. * white_value / g_w
    b_coeff = 1. * white_value / b_w

    image_wb[:, :, 0] = np.minimum(image_wb[:, :, 0] * r_coeff, 255).astype('uint8')
    image_wb[:, :, 1] = np.minimum(image_wb[:, :, 1] * g_coeff, 255).astype('uint8')
    image_wb[:, :, 2] = np.minimum(image_wb[:, :, 2] * b_coeff, 255).astype('uint8')

    return image_wb


def adjust_white_balance(image):
    ref_rect = get_reference_rect(image)
    base_white_rect = image[ref_rect[1]: ref_rect[3], ref_rect[0]: ref_rect[2], :]
    rw = base_white_rect[:, :, 0].mean()
    gw = base_white_rect[:, :, 1].mean()
    bw = base_white_rect[:, :, 2].mean()
    image_wb = adjust_white_balance_wrt_rgb(image, (rw, bw, gw))
    return image_wb


def get_pool_boxes(image):
    """
    Take an image (RGB or grayscale, whatever) of stripped test stripe and return array of boxes (x1, y1, x2, y2) 
    corresponding to every pool on a strip.
    """

    a, b = image.shape[:2]
    k = a / STRIPE_A

    boxes = np.array([[k * (LEFT_MARGIN + (POOL_WIDTH + GAP_WIDTH) * j), 0,
                       k * (LEFT_MARGIN + (POOL_WIDTH + GAP_WIDTH) * j + POOL_WIDTH), a]
                      for j in xrange(12)], dtype=np.uint16)

    return boxes
