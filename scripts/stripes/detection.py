# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Pipeline for detection of a single stripe on image
#
# Steps:
# 1. Make a grayscale copy
# 2. Blur it, apply adaptive thresholding such that stripe rectangle is easily distinguishable from background
# 3. Find largest contour (~ our stripe)
# 4. Find bounding rectangle for it
# 5. Apply perspective transformation and crop
# 6. Return crop
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import division
import cv2
import numpy as np


class Detector:

    def __init__(self):

        # images ~ intermediate steps
        self.image_source = None
        self.image_binary = None
        self.warp_matrix = None
        self.image_warped = None

        # magic numbers
        self.height = 480
        self.stripes_num = 20.
        self.stripes_size = 100.
        self.d_bil_fil = 50
        self.sigma_x_bil_fil = 50
        self.sigma_c_bil_fil = 25
        self.thres_block = 100
        self.thres_c = 0
        self.target_rect = [0, 0, 25 * 24, 25]

    @staticmethod
    def odd(x):
        """
        Check if x is odd, make x odd if not.
        """
        if x % 2 == 0:
            x += 1
        return x

    @staticmethod
    def resize(image, size):
        """
        Resize image to given size (if size is tuple) or adjust size such that image height is "size" (if size is int)
        """
        if isinstance(size, tuple):
            if len(size) == 2:
                return cv2.resize(image, size)
            else:
                raise ValueError("Weird size tuple")
        elif isinstance(size, int):
            h, w = image.shape[:2]

            dwidth = int((1. * size / h) * w)
            return cv2.resize(image, (dwidth, size))

    @staticmethod
    def get_largest_contour_id(contours):
        """
        Get index of contour with max area among all contours.
        """
        assert len(contours) > 0, "Empty contour"
        max_area = -1
        max_id = -1
        for j in xrange(len(contours)):
            area = cv2.contourArea(contours[j])
            if area > max_area:
                max_area = area
                max_id = j
        return max_id

    @staticmethod
    def get_rect(contour):
        """
        Get coordinates of bounding rotated rect for contour. 
        """
        bounding_rot_rect = cv2.minAreaRect(contour)
        verts = cv2.boxPoints(bounding_rot_rect)
        return verts.astype(np.uint)

    @staticmethod
    def get_correct_arrangement(rect):
        """
        Find arrangment indices as follows
        # 0 -> 1
        # |    |
        # 3 <- 2
        """
        arrangement = [-1] * 4
        x1i, x2i, x3i, x4i = np.argsort(rect[:, 0])

        if rect[x1i, 1] > rect[x2i, 1]:
            arrangement[0] = x2i
            arrangement[3] = x1i
        else:
            arrangement[0] = x1i
            arrangement[3] = x2i

        if rect[x3i, 1] > rect[x4i, 1]:
            arrangement[1] = x4i
            arrangement[2] = x3i
        else:
            arrangement[2] = x4i
            arrangement[1] = x3i

        return arrangement

    def binarize(self, image):
        """
        Binarize image with flexible parameters.
        """
        # convert if necessary
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 2:
            image_gray = image
        else:
            raise ValueError("Weird shape of image")

        # get shape parameters and calculate size-dependent coefficient
        height, width = image.shape[:2]
        coeff = (height / self.stripes_num) / self.stripes_size

        # adjust filtering parameters
        d = self.odd(int(self.d_bil_fil * coeff))
        sigma_x = self.odd(int(self.sigma_x_bil_fil * coeff))
        sigma_c = self.sigma_c_bil_fil

        # adjust thresholding parameters
        block = self.odd(int(self.thres_block * coeff))
        c = self.thres_c

        # adjust morphology parameters
        # pass

        # apply
        image_blurred = cv2.bilateralFilter(image_gray, d, sigma_c, sigma_x)
        image_binary = cv2.adaptiveThreshold(image_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                             block, c)

        return image_binary

    def get_homography(self, src_rect):
        """
        Calculate matrix for perspective warping that transforms src_rect into dist_rect.
        """
        # coordinates of final crop
        dist_rect = np.array([[self.target_rect[0], self.target_rect[1]],
                              [self.target_rect[2], self.target_rect[1]],
                              [self.target_rect[2], self.target_rect[3]],
                              [self.target_rect[0], self.target_rect[3]]])

        # rearrange the vertices to correspond with dist_rect
        arrangement = self.get_correct_arrangement(src_rect)
        src_rect = src_rect[arrangement]

        # add dimensiong (cv2 feature)
        src_rect = np.expand_dims(src_rect, 1)
        dist_rect = np.expand_dims(dist_rect, 1)

        # get matrix for warp
        h, status = cv2.findHomography(src_rect, dist_rect)
        return h

    def get_warp_matrix(self, image):
        """
        Find max-area contour on binarized image, fit into rotated rect and find matrix to warp it to self.target_rect
        """
        assert len(image.shape) == 2, "Not a grayscale image"

        # get all contours, choose the one with max area
        _, contours, __ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_id = self.get_largest_contour_id(contours)

        # get vertices of bounding rotated rect
        rect = self.get_rect(contours[max_id])

        # find matrix for warp
        h = self.get_homography(rect)
        return h

    def warp(self, image, warp_matrix):
        """
        Apply perspective warp to image.
        """
        return cv2.warpPerspective(image, warp_matrix, (self.target_rect[2], self.target_rect[3]))

    def detect(self, image):
        """
        Apply complete pipeline to image.
        """
        # resize
        self.image_source = self.resize(image, self.height)

        # binarize
        self.image_binary = self.binarize(self.image_source)

        # warp
        self.warp_matrix = self.get_warp_matrix(self.image_binary)
        self.image_warped = self.warp(self.image_source, self.warp_matrix)

        # you're beatiful
        return self.image_warped





