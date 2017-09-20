import cv2
import numpy as np


class Table:
    def __init__(self, scale, w_template_mm, h_template_mm, w_roi_mm, h_roi_mm,
                 w_strip_mm, h_strip_mm, coords_roi_mm, coords_strip_mm):
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

        self.template = None  # picture of whole template
        self.roi = None  # picture of ROI (no borders)
        self.strip = None  # picture of strip (extracted)
        self.palette = None  # dictionary with support colors palette
        self.colorbar = None  # dictionary with strip colors data

        self.coords_roi_px = {}  # dict of calibration pools coordinates (in px) (on roi)
        self.coords_strip_px = {}  # dict of calibration pools coordinates (in px) (on strip)
        self._rescale_coords_()

    def _rescale_coords_(self):
        """
        Rescale and update coords dicts and attributes (from mm to pixels via scaling).
        """

        self.w_template_px = self.w_template_mm * self.scale
        self.h_template_px = self.h_template_mm * self.scale
        self.w_roi_px = self.w_roi_mm * self.scale
        self.h_roi_px = self.h_roi_mm * self.scale
        self.w_strip_px = self.w_strip_mm * self.scale
        self.h_strip_px = self.h_strip_mm * self.scale

        if self.coords_roi_mm is None:
            raise ValueError("Nothing to rescale, coords_roi_mm not found")
        elif self.coords_strip_mm is None:
            raise ValueError("Nothing to rescale, coords_strip_mm not found")
        for key, rect in self.coords_roi_mm.iteritems():
            self.coords_roi_px[key] = ((int(rect[0][0] * self.scale), int(rect[0][1] * self.scale)),
                                       (int(rect[1][0] * self.scale), int(rect[1][1] * self.scale)))

        for key, rect in self.coords_strip_mm.iteritems():
            self.coords_strip_px[key] = ((int(rect[0][0] * self.scale), int(rect[0][1] * self.scale)),
                                         (int(rect[1][0] * self.scale), int(rect[1][1] * self.scale)))

    # private methods for image preprocessing:

    @staticmethod
    def _pyr_down_(image, n=1):
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
        return self._pyr_down_(image, n=n)

    @staticmethod
    def _contrast_(image):
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            return cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)), cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _denoise_(image, ksize=15):
        return cv2.medianBlur(image, ksize=ksize)

    @staticmethod
    def _binarize_(image, blocksize=11, c=0):
        if len(image.shape) == 2:
            out = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                        blockSize=blocksize, C=c)
        else:
            out = cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, blockSize=blocksize, C=c)
        return out

    @staticmethod
    def _morphology_open_(image, ksize=3):
        kernel = np.ones(ksize)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def _morphology_close_(image, ksize=3):
        kernel = np.ones(ksize)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # private methods for contours handling:

    @staticmethod
    def _get_max_area_contour_id_(contours):
        max_area = 0.
        max_contour_id = -1
        for j, contour in enumerate(contours):
            new_area = cv2.contourArea(contour)
            if new_area >= max_area:
                max_area = new_area
                max_contour_id = j
        return max_contour_id

    @staticmethod
    def _approximate_contour_(contour, eps=5 * 1e-3, closed=True):
        return cv2.approxPolyDP(contour, epsilon=eps * cv2.arcLength(contour, closed=closed), closed=closed)

    @staticmethod
    def _get_correct_arrangement_(rect):
        """
        Find arrangment indices of :rect:-vertices as follows (clockwise, from top-left): 0 -> 1 -> 2 -> 3
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

    @staticmethod
    def _check_contour_(contour, eps_area, eps_cos=None):
        if cv2.contourArea(contour) < eps_area:
            return False  # contour too small
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) / min(w, h) > 100:
            return False  # contour too narrow
        # TODO: Similarity to rectangle (for all angles abs(cos) < eps_cos)
        return True

    # private methods for image transforming:

    @staticmethod
    def _get_warp_matrix_(src_rect, dist_rect):
        assert src_rect.shape == dist_rect.shape, "Shapes mismatch: {} != {}".format(src_rect.shape, dist_rect.shape)
        h, status = cv2.findHomography(src_rect, dist_rect)
        return h

    @staticmethod
    def _warp_(image, warp_matrix, dist_shape=None):
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
            print "Empty image"
            return None
        image_denoised = self._denoise_(image)
        image_binary = self._binarize_(image_denoised)
        image_morph = self._morphology_open_(image_binary)

        r, contours, h = cv2.findContours(image_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_id = self._get_max_area_contour_id_(contours)
        contour = contours[max_id]
        poly = self._approximate_contour_(contour)
        if len(poly) != 4:
            print "Failed to approx with 4 points"
            return poly

        correct_arrangement = self._get_correct_arrangement_(np.squeeze(poly, 1))
        src_rect = np.squeeze(poly, 1)[correct_arrangement]
        dist_rect = np.array([[[0, 0]],
                              [[self.w_template_px, 0]],
                              [[self.w_template_px, self.h_template_px]],
                              [[0, self.h_template_px]]])
        warp_matrix = self._get_warp_matrix_(np.expand_dims(src_rect, 1), dist_rect)
        template = self._warp_(image, warp_matrix, dist_shape=(self.w_template_px, self.h_template_px))
        if return_binary:
            return template, image_morph
        return template

    def _detect_roi_(self, template):
        """
        Detect rectangle inside :template: and return its warp-perspective transformed copy.
        """
        if template is None:
            print "Empty template"
            return None

        edges = cv2.Canny(template, 100, 200)
        _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        poly = self._approximate_contour_(contour)
        if len(poly) != 4:
            print "Failed to approx with 4 points"
            return None

        correct_arrangement = self._get_correct_arrangement_(np.squeeze(poly, 1))
        src_rect = np.squeeze(poly, 1)[correct_arrangement]
        dist_rect = np.array([[[0, 0]],
                              [[self.w_roi_px, 0]],
                              [[self.w_roi_px, self.h_roi_px]],
                              [[0, self.h_roi_px]]])
        warp_matrix = self._get_warp_matrix_(np.expand_dims(src_rect, 1), dist_rect)
        roi = self._warp_(template, warp_matrix, dist_shape=(self.w_roi_px, self.h_roi_px))
        return roi

    def _detect_strip_(self, roi, return_binary=False):
        """
        Detect strip rectangle inside :roi: and return its warp-perspective transformed copy.
        """
        if roi is None:
            print "Empty ROI"
            return None

        strip_rect = roi[self.coords_roi_px['strip'][0][1]: self.coords_roi_px['strip'][1][1],
                         self.coords_roi_px['strip'][0][0]: self.coords_roi_px['strip'][1][0], :]
        strip_rect_denoised = self._denoise_(strip_rect, ksize=9)
        # strip_rect_binary = self._binarize_(strip_rect_denoised, blocksize=101, c=0)
        strip_rect_binary = self._binarize_(strip_rect_denoised, blocksize=strip_rect.shape[1] / 5, c=0)
        strip_rect_morph = self._denoise_(strip_rect_binary, ksize=3)
        strip_rect_morph = self._denoise_(strip_rect_morph, ksize=3)
        # strip_rect_morph = self._morphology_close_(strip_rect_morph, ksize=3)

        r, contours, h = cv2.findContours(strip_rect_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_id = self._get_max_area_contour_id_(contours)
        # max_contour = contours[max_id]
        # good_contours = [max_contour]
        good_contours = []

        for cont in contours:
            if self._check_contour_(contour=cont, eps_area=((strip_rect.shape[0] / 3) ** 2)):
                good_contours.append(cont)
        contour = reduce(lambda x, y: np.concatenate((x, y)), good_contours)
        poly = self._approximate_contour_(contour, eps=2 * 1e-3, closed=False)

        rotated_rect = cv2.minAreaRect(poly)
        src_points = cv2.boxPoints(rotated_rect).astype(np.int)
        # cv2.drawContours(strip_rect_contoured, [src_points], 0, (255, 255, 255), 1)
        correct_arrangement = self._get_correct_arrangement_(src_points)
        src_rect = np.expand_dims(src_points[correct_arrangement], 1)
        dist_rect = np.array([[[0, 0]],
                              [[self.w_strip_px, 0]],
                              [[self.w_strip_px, self.h_strip_px]],
                              [[0, self.h_strip_px]]])

        warp_matrix = self._get_warp_matrix_(src_rect, dist_rect)
        strip = self._warp_(strip_rect, warp_matrix, dist_shape=(self.w_strip_px, self.h_strip_px))
        if return_binary:
            return strip, strip_rect_morph
        return strip

    # private methods for reading colors:

    def _read_palette_(self, roi):
        if roi is None:
            raise ValueError("No template found")
        elif self.coords_roi_px is None:
            raise ValueError("No coordinate data found")
        else:
            palette = {key: roi[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0], :] for key, rect in
                       self.coords_roi_px.iteritems()}
        return palette

    def _read_colorbar_(self, strip):
        if strip is None:
            raise ValueError("No strip found")
        elif self.coords_strip_px is None:
            raise ValueError("No coordinate data found")
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
