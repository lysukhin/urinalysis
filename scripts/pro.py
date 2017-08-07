from stripes.detection import Detector
from stripes.preprocessing import adjust_white_balance, get_pool_boxes
from stripes.analysis import calculate_pro

import matplotlib.pyplot as plt


def pro(image):
    """
    Calculate PRO content using image (cropped to rectangle of stripe).
    Pool box with PRO test is 3rd (2nd in zero indexing) from the left. 
    """

    # Adjust white balance
    image_wb = adjust_white_balance(image)

    # Get (hard-coded) coords of test pools
    pool_boxes = get_pool_boxes(image_wb)

    # Choose PRO pool box
    pro_pool_box = pool_boxes[2]

    # Crop box from image
    pro_pool_crop = image_wb[pro_pool_box[1]: pro_pool_box[3], pro_pool_box[0]: pro_pool_box[2], :]
    plt.imshow(pro_pool_crop, cmap='gray')
    plt.show()

    # Get channels means
    red = pro_pool_crop[:, :, 0].mean()
    green = pro_pool_crop[:, :, 1].mean()
    blue = pro_pool_crop[:, :, 2].mean()

    # Calculate PRO content via linear regression
    # NB: may be negative!
    pro_content = calculate_pro(red, green, blue)

    return pro_content
