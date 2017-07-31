# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Pipeline for detection of a single stripe on image
#
# Steps:
# 1. Make a grayscale, contrast-enhanced copy of image
# 2. Find edges, apply morphological operations to clean up
# 3. Find all contours of edges-image, merge them to one
# 4. Find bounding rectangle for merged contour
# 5. Return crop with this bounding rectangle
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import cv2
import numpy as np
import matplotlib.pyplot as plt


class StripeDetector:

    def __init__(self):

        self.src = None
        self.gray = None
        self.gray_enh = None
        self.edges = None
        self.edges_clean = None
        self.contour = None
        self.contours = None
        self.brect = None
        self.crop = None

        self.CANNY_THRES_LO = 50
        self.CANNY_THRES_HI = 250
        self.CANNY_APERTURE = 3

        self.MORPH_KERNEL_SIZE = (7, 7)

    def get_gray(self, image=None):
        if image is None:
            image = self.src
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.gray

    def get_gray_enh(self, image=None):
        if image is None:
            image = self.gray
        self.gray_enh = cv2.equalizeHist(image)
        return self.gray_enh

    def get_edges(self, image=None):
        if image is None:
            image = self.gray_enh
        self.edges = cv2.Canny(image, threshold1=self.CANNY_THRES_LO,
                               threshold2=self.CANNY_THRES_HI,
                               apertureSize=self.CANNY_APERTURE,
                               L2gradient=True)
        return self.edges

    def clean_edges(self, edges=None):
        if edges is None:
            image = self.edges
        kernel = np.ones(self.MORPH_KERNEL_SIZE, np.uint8)
        self.edges_clean = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return self.edges_clean

    def get_contours(self, edges=None):
        if edges is None:
            image = self.edges_clean
        _, self.contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def union_contours(self, contours=None, shape=None):
        if contours is None:
            contours = self.contours
            shape = self.edges_clean.shape
        tmp_img = np.zeros(shape)
        for j in xrange(len(contours)):
            cv2.drawContours(tmp_img, contours, j, 255, -1)
        tmp_img = tmp_img.astype(np.uint8)
        kernel = np.ones(self.MORPH_KERNEL_SIZE, np.uint8)
        image_filled = cv2.morphologyEx(tmp_img, cv2.MORPH_OPEN, kernel)
        __, contours, hierarchy2 = cv2.findContours(image_filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contour = contours[0]
        return self.contour

    def get_bound_rect(self, contour=None):
        if contour is None:
            contour = self.contour
        self.brect = cv2.boundingRect(contour)
        return self.brect

    def get_crop(self, image=None, brect=None):
        if image is None:
            image = self.src.copy()
        if brect is None:
            brect = self.brect
        self.crop = image[brect[1]: brect[1] + brect[3], brect[0]: brect[0] + brect[2], :]
        return self.crop

    def detect(self, image):
        assert image.shape[-1] == 3, "Not a 3-channel image"
        self.src = image
        self.get_gray()
        self.get_gray_enh()
        self.get_edges()
        self.clean_edges()
        self.get_contours()
        self.union_contours()
        self.get_bound_rect()
        self.get_crop()
        return self.crop

    def show_steps(self):

        contoured = self.src.copy()
        for j in xrange(len(self.contours)):
            cv2.drawContours(contoured, self.contours, j, (255, 0, 0), 2)
        cv2.drawContours(contoured, [self.contour], 0, (0, 255, 0), 2)

        imgs = [self.src, self.gray, self.gray_enh, self.edges, self.edges_clean,
                contoured, self.crop]
        titles = ["Source", "Gray", "Gray HEq", "Edges", "Edges cleaned", "Contours", "Crop"]
        cmaps = [None, "gray", "gray", "gray", "gray", "gray", None, None]

        for j, (img, title, cmap) in enumerate(zip(imgs, titles, cmaps), 1):

            plt.title(title)
            plt.imshow(img, cmap=cmap)
            plt.show()