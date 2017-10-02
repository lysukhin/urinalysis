from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Table:
    def __init__(self, scale, w_template_mm, h_template_mm, w_roi_mm, h_roi_mm,
                 w_strip_mm, h_strip_mm, coords_roi_mm, coords_strip_mm,
                 descriptor=cv2.AKAZE_create(), matcher_type=cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING):
        """
        Initialize Table class instance with following parameters:
        :param scale: 			# scaling factor for mm->pixels conversion
        :param w_template_mm:	# size of template (width)
        :param h_template_mm:	# size of template (height)
        :param w_roi_mm:		# size of roi (width)
        :param h_roi_mm:		# size of roi (height)
        :param w_strip_mm:		# size of strip (width)
        :param h_strip_mm:		# size of strip (height)
        :param coords_roi_mm:	# dict of calibration pools coordinates (in mm) (on roi)
        :param coords_strip_mm:	# dict of calibration pools coordinates (in mm) (on strip)
        :param descriptor: # initializer for keypoints and descriptors extractor 
        :param matcher_type:    # type of matcher passed to cv2.DescriptorMatcher_create()
        """
        self.scale = scale
        self.w_template_mm = w_template_mm
        self.h_template_mm = h_template_mm
        self.w_roi_mm = w_roi_mm
        self.h_roi_mm = h_roi_mm
        self.w_strip_mm = w_strip_mm
        self.h_strip_mm = h_strip_mm
        self.coords_roi_mm = coords_roi_mm
        self.coords_strip_mm = coords_strip_mm
        self.descriptor = descriptor
        self.matcher = cv2.DescriptorMatcher_create(matcher_type)

        self.template = None  # picture of whole template
        self.roi = None  # picture of ROI (no borders)
        self.strip = None  # picture of strip (extracted)
        self.palette = None  # dictionary with support colors palette
        self.colorbar = None  # dictionary with strip colors data

        self.coords_roi_px = {}  # dict of calibration pools coordinates (in px) (on roi)
        self.coords_strip_px = {}  # dict of calibration pools coordinates (in px) (on strip)
        self._rescale_coords_()  # rescale coordinates from mm to px

    def _rescale_coords_(self):
        """
        Rescale and update coords dicts and attributes (from mm to pixels via scaling).
        """

        self.w_template_px = int(self.w_template_mm * self.scale)
        self.h_template_px = int(self.h_template_mm * self.scale)
        self.w_roi_px = int(self.w_roi_mm * self.scale)
        self.h_roi_px = int(self.h_roi_mm * self.scale)
        self.w_strip_px = int(self.w_strip_mm * self.scale)
        self.h_strip_px = int(self.h_strip_mm * self.scale)

        if self.coords_roi_mm is None:
            raise AttributeError("Nothing to rescale, coords_roi_mm not found")
        elif self.coords_strip_mm is None:
            raise AttributeError("Nothing to rescale, coords_strip_mm not found")
        for key, rect in self.coords_roi_mm.iteritems():
            self.coords_roi_px[key] = ((int(rect[0][0] * self.scale), int(rect[0][1] * self.scale)),
                                       (int(rect[1][0] * self.scale), int(rect[1][1] * self.scale)))

        for key, rect in self.coords_strip_mm.iteritems():
            self.coords_strip_px[key] = ((int(rect[0][0] * self.scale), int(rect[0][1] * self.scale)),
                                         (int(rect[1][0] * self.scale), int(rect[1][1] * self.scale)))

    # methods for image preprocessing:

    @staticmethod
    def odd(x):
        return x + int(x % 2 == 0)

    @staticmethod
    def pyr_down(image, n=1):
        """
        pyrDown :image: :n: times.
        """
        if n <= 1:
            return image
        shrinked = image.copy()
        for time in xrange(n):
            shrinked = cv2.pyrDown(shrinked)
        return shrinked

    def _downsample_(self, image, to_size=960):
        """
        Resize image such that larger side is of :to_size: scale.
        """
        n = int(np.log2(max(image.shape) / to_size))
        return self.pyr_down(image, n=n)

    @staticmethod
    def contrast(image):
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            return cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)), cv2.COLOR_GRAY2RGB)

    @staticmethod
    def denoise(image, ksize=15):
        return cv2.medianBlur(image, ksize=ksize)

    @staticmethod
    def binarize(image, blocksize=11, c=0):
        if len(image.shape) == 2:
            out = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                        blockSize=blocksize, C=c)
        else:
            out = cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, blockSize=blocksize, C=c)
        return out

    @staticmethod
    def morphology_open(image, ksize=3):
        kernel = np.ones(ksize)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def morphology_close(image, ksize=3):
        kernel = np.ones(ksize)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # methods for contours handling:

    @staticmethod
    def get_max_area_contour_id(contours):
        max_area = 0.
        max_contour_id = -1
        for j, contour in enumerate(contours):
            new_area = cv2.contourArea(contour)
            if new_area >= max_area:
                max_area = new_area
                max_contour_id = j
        return max_contour_id

    @staticmethod
    def approximate_contour(contour, eps=5 * 1e-3, closed=True):
        return cv2.approxPolyDP(contour, epsilon=eps * cv2.arcLength(contour, closed=closed), closed=closed)

    @staticmethod
    def get_correct_arrangement(rect):
        """
        Find arrangment indices of :rect:-vertices as follows (clockwise, from top-left): 0 -> 1 -> 2 -> 3
        """
        arrangement = [-1] * 4
        rect = np.array(rect)
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

    @staticmethod
    def check_contour(contour, eps_area):
        if cv2.contourArea(contour) < eps_area:
            return False  # contour too small
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) / min(w, h) > 100:
            return False  # contour too narrow
        return True

    # methods for keypoint detection and matching

    def _get_keypoints_and_descriptors_(self, image):
        if self.descriptor is None:
            raise AttributeError("No descriptor class instantiated")
        keypoints, descriptors = self.descriptor.detectAndCompute(image, mask=None)
        return keypoints, descriptors

    def _get_good_matches_(self, descriptors1, descriptors2, matching_coeff=0.25):
        if self.descriptor is None:
            raise AttributeError("No matcher class instantiated")
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < matching_coeff * n.distance:
                good_matches.append(m)
        return good_matches

    def _get_keypoints_matches_(self, keypoints1, keypoints2, good_matches):
        """
        :param keypoints1: query keypoints 
        :param keypoints2: scene keypoints
        :param good_matches: 
        :return: 
        """
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        return src_pts, dst_pts

    # methods for image transforming:

    @staticmethod
    def get_warp_matrix(src_rect, dist_rect):
        assert src_rect.shape == dist_rect.shape, "Shapes mismatch: {} != {}".format(src_rect.shape, dist_rect.shape)
        h, status = cv2.findHomography(src_rect, dist_rect, cv2.RANSAC)#, 5.0)
        return h

    @staticmethod
    def warp(image, warp_matrix, dist_shape=None):
        if dist_shape is None:
            dist_shape = image.shape[:2]
        warped = cv2.warpPerspective(src=image, M=warp_matrix, dsize=dist_shape)
        return warped

    # private methods for object detection:

    def _detect_template_(self, image, return_binary=False):
        """
        Detect white 4-vertices white polygon on :image:, return its warp-perspective transformed copy.
        """
        if image is None:
            raise AttributeError("No image to detect template on")

        image_denoised = self.denoise(image)
        image_binary = self.binarize(image_denoised, blocksize=self.odd(image.shape[1] // 5))
        image_morph = self.morphology_open(image_binary, ksize=self.odd(image.shape[1] // 250))
        if return_binary:
            return image_morph
        r, contours, h = cv2.findContours(image_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_id = self.get_max_area_contour_id(contours)
        contour = contours[max_id]
        poly = None
        for eps in [1e-4, 5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2]:
            poly_eps = self.approximate_contour(contour, eps=eps)
            if poly_eps.shape[0] == 4:
                poly = poly_eps.copy()
        if poly is None:
            raise RuntimeError("Failed to approx template with 4 points")

        correct_arrangement = self.get_correct_arrangement(np.squeeze(poly, 1))
        src_pts_verts = poly[correct_arrangement]
        dst_pts_verts = np.array([[[0, 0]],
                                  [[self.w_template_px, 0]],
                                  [[self.w_template_px, self.h_template_px]],
                                  [[0, self.h_template_px]]])

        # keypoints stuff
        if self.descriptor is not None:
            reference_template = cv2.imread("/data/Y.Disk/work/urinalysis/scripts/table/template.png",
                                            cv2.IMREAD_COLOR)
            reference_template = cv2.cvtColor(cv2.resize(reference_template, (self.w_template_px, self.h_template_px)),
                                              cv2.COLOR_BGR2RGB)

            keypoints_query, descriptors_query = self._get_keypoints_and_descriptors_(reference_template)
            keypoints_scene, descriptors_scene = self._get_keypoints_and_descriptors_(image)
            good_matches = self._get_good_matches_(descriptors_query, descriptors_scene)
            src_pts_kp, dst_pts_kp = self._get_keypoints_matches_(keypoints_query, keypoints_scene, good_matches)
            src_pts = np.concatenate((src_pts_verts, src_pts_kp), axis=0)
            dst_pts = np.concatenate((dst_pts_verts, dst_pts_kp), axis=0)
            # for pt in src_pts:
            #     cv2.circle(image, tuple(pt.astype(np.int).ravel().tolist()), 6, (0, 255, 255), -1)
            # cv2.drawKeypoints(reference_template, keypoints_query, reference_template)
        else:
            src_pts = src_pts_verts
            dst_pts = dst_pts_verts

        warp_matrix = self.get_warp_matrix(src_pts, dst_pts)
        template = self.warp(image, warp_matrix, dist_shape=(self.w_template_px, self.h_template_px))
        # if return_binary:
        #     return template, image_morph
        # else:
        return template

    def _detect_roi_(self, template, return_contour=False):
        """
        Detect rectangle inside :template: and return its warp-perspective transformed copy.
        """
        if template is None:
            raise AttributeError("No template to detect ROI on")

        edges = cv2.Canny(template, 100, 200)
        _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        poly = None
        for eps in [1e-4, 5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2][::-1]:
            poly_eps = self.approximate_contour(contour, eps=eps)
            if poly_eps.shape[0] == 4:
                poly = poly_eps.copy()
        if poly is None:
            raise RuntimeError("Failed to approx roi with 4 points")

        correct_arrangement = self.get_correct_arrangement(np.squeeze(poly, 1))
        src_pts_verts = poly[correct_arrangement]
        dst_pts_verts = np.array([[[0, 0]],
                                  [[self.w_roi_px, 0]],
                                  [[self.w_roi_px, self.h_roi_px]],
                                  [[0, self.h_roi_px]]])

        # keypoints stuff
        if self.descriptor is not None:
            reference_roi = cv2.imread("/data/Y.Disk/work/urinalysis/scripts/table/roi.png",
                                       cv2.IMREAD_COLOR)
            reference_roi = cv2.cvtColor(cv2.resize(reference_roi, (self.w_roi_px, self.h_roi_px)),
                                         cv2.COLOR_BGR2RGB)

            keypoints_query, descriptors_query = self._get_keypoints_and_descriptors_(reference_roi)
            keypoints_scene, descriptors_scene = self._get_keypoints_and_descriptors_(template)
            good_matches = self._get_good_matches_(descriptors_query, descriptors_scene)
            src_pts_kp, dst_pts_kp = self._get_keypoints_matches_(keypoints_query, keypoints_scene, good_matches)
            src_pts = np.concatenate((src_pts_verts, src_pts_kp), axis=0)
            dst_pts = np.concatenate((dst_pts_verts, dst_pts_kp), axis=0)
            # for pt in src_pts:
            #     cv2.circle(template, tuple(pt.astype(np.int).ravel().tolist()), 6, (0, 255, 255), -1)
            # cv2.drawKeypoints(reference_roi, keypoints_query, reference_roi)
        else:
            src_pts = src_pts_verts
            dst_pts = dst_pts_verts

        warp_matrix = self.get_warp_matrix(src_pts, dst_pts)
        roi = self.warp(template, warp_matrix, dist_shape=(self.w_roi_px, self.h_roi_px))

        if return_contour:
            contoured = template.copy()
            cv2.drawContours(contoured, [poly], 0, (255, 255, 255), 2)
            return roi, contoured
        else:
            return roi

    def _detect_strip_(self, roi, return_binary=False, return_poly=False, return_contour=False):
        """
        Detect strip rectangle inside :roi: and return its warp-perspective transformed copy.
        """
        if roi is None:
            raise AttributeError("No ROI to detect strip on")

        strip_rect = roi[self.coords_roi_px['strip'][0][1]: self.coords_roi_px['strip'][1][1],
                         self.coords_roi_px['strip'][0][0]: self.coords_roi_px['strip'][1][0], :]
        strip_rect_denoised = self.denoise(strip_rect, ksize=9)
        strip_rect_binary = self.binarize(strip_rect_denoised, blocksize=self.odd(strip_rect.shape[1] // 5), c=0)
        strip_rect_morph = self.denoise(strip_rect_binary, ksize=5)
        strip_rect_morph = self.denoise(strip_rect_morph, ksize=5)
        strip_rect_morph = self.morphology_open(strip_rect_morph, ksize=21)

        cv2.rectangle(strip_rect_morph, (0, 0), strip_rect_morph.shape[::-1], (0, 0, 0), 1)

        r, contours, h = cv2.findContours(strip_rect_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        good_contours = []

        for cont in contours:
            if self.check_contour(contour=cont, eps_area=((strip_rect.shape[0] / 4) ** 2)):
                good_contours.append(cont)
        contour = reduce(lambda x, y: np.concatenate((x, y)), good_contours)

        rotated_rect = cv2.minAreaRect(contour)
        src_points = cv2.boxPoints(rotated_rect).astype(np.int)
        correct_arrangement = self.get_correct_arrangement(src_points)
        src_rect = src_points[correct_arrangement]
        dist_rect = np.array([[[0, 0]],
                              [[self.w_strip_px, 0]],
                              [[self.w_strip_px, self.h_strip_px]],
                              [[0, self.h_strip_px]]])
        warp_matrix = self.get_warp_matrix(np.expand_dims(src_rect, 1), dist_rect)

        strip = self.warp(strip_rect, warp_matrix, dist_shape=(self.w_strip_px, self.h_strip_px))
        output = strip
        # if return_binary:
        #     output.append(strip_rect_morph)
        # if return_poly:
        #     output.append(src_points)
        # if return_contour:
        #     output.append(contour)
        return output

    # private methods for reading colors:

    def _read_palette_(self, roi):
        if roi is None:
            raise AttributeError("No template found")
        elif self.coords_roi_px is None:
            raise AttributeError("No coordinate data found")
        else:
            palette = {key: roi[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0], :] for key, rect in
                       self.coords_roi_px.iteritems()}
        return palette

    def _read_colorbar_(self, strip):
        if strip is None:
            raise AttributeError("No strip found")
        elif self.coords_strip_px is None:
            raise AttributeError("No coordinate data found")
        else:
            colorbar = {key: strip[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0], :] for key, rect in
                        self.coords_strip_px.iteritems()}
        return colorbar

    # public methods

    def fit(self, image):
        """
        Runs detection of all necessary parts (template, roi, strip) on :image:, 
        stores results in class fields and returns strip.
        """
        if image is None:
            print "Empty image"
            return None

        downsampled = self._downsample_(image)
        self.template = self._detect_template_(downsampled)
        self.roi = self._detect_roi_(self.template)
        self.strip = self._detect_strip_(self.roi)

        self.colorbar = self._read_colorbar_(self.strip)
        self.palette = self._read_palette_(self.roi)

        return self
