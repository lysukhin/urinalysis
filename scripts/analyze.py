import os
import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stripes.preprocessing as preproc
# import stripes.analysis as analysis
# import stripes.palette as palette
from stripes.detection import StripeDetector

plt.rcParams['figure.figsize'] = (15, 5)
sns.set_style('white')


def dump(fname, imname, values):
    # {pool_i : [R, G, B], ...}
    line = [values[pool] for pool in sorted(values.keys())]
    line = reduce(lambda x, y: x + y, line)
    line = map(str, line)
    line_str = ";".join(line)
    with open(fname, 'a') as f:
        f.write(imname + ';' + line_str + "\n")


def main():

    # Read image from command-line
    if len(sys.argv) < 2:
        raise ValueError("Path to image not specified")
    elif not os.path.isfile(sys.argv[1]):
        raise ValueError("Image not found")
    else:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect strip
    detector = StripeDetector()
    crop = detector.detect(image)
    plt.imshow(detector.show_steps())
    plt.show()

    # Adjust wb
    crop_wb = preproc.adjust_white_balance(crop)

    # Get color pools
    crop_wb = cv2.resize(crop_wb, (25 * 24, 25))
    pool_boxes = preproc.get_pool_boxes(crop_wb)

    values = {}
    for j, pool_box in enumerate(pool_boxes, 1):
        pool = crop_wb[:, pool_box[0]: pool_box[2], :]
        red = pool[:, :, 0]
        green = pool[:, :, 1]
        blue = pool[:, :, 2]

        # plt.subplot(1, 2, 1)
        # plt.imshow(pool)
        # plt.subplot(1, 2, 2)
        # plt.hist(red.ravel(), bins=256 / 4, normed=True, color='r', alpha=0.3, label="Mean = {}".format(red.mean()))
        # plt.hist(green.ravel(), bins=256 / 4, normed=True, color='g', alpha=0.3, label="Mean = {}".format(green.mean()))
        # plt.hist(blue.ravel(), bins=256 / 4, normed=True, color='b', alpha=0.3, label="Mean = {}".format(blue.mean()))
        # plt.legend()
        # plt.show()

        values.update({j: [red.mean(), red.std(), green.mean(), green.std(), blue.mean(), blue.std()]})

    dump("colors.csv", sys.argv[1], values)

if __name__ == "__main__":
    main()




