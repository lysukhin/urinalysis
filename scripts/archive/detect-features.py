import cv2
import glob
import numpy as np

def pyr_down(image, n=1):
    if n <= 1:
        return image
    shrinked = image.copy()
    for time in xrange(n):
        shrinked = cv2.pyrDown(shrinked)
    return shrinked

def shrink(image, to_size=500):
    n = int(np.log2(max(image.shape) / to_size))
    return pyr_down(image, n=n)

def match_image(scene_image, query_image):
    descr = cv2.AKAZE_create()
    keypoints_query, descriptors_query = descr.detectAndCompute(query_image, mask=None)
    keypoints_scene, descriptors_scene = descr.detectAndCompute(scene_image, mask=None)
    MIN_MATCH_COUNT = 5
    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.knnMatch(descriptors_query, descriptors_scene, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)           
        
    if len(good) > MIN_MATCH_COUNT:
        print "Enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)

        src_pts = np.float32([keypoints_query[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = query_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        scene_image = cv2.polylines(scene_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # cv2.imshow("scene", scene_image)
    else:
        print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None
        
    draw_params = dict(matchColor = (255, 255, 0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    matched_image = cv2.drawMatches(query_image, keypoints_query, scene_image, keypoints_scene, good, None, **draw_params)

    return matched_image

def main():
	
	query_image = cv2.flip(cv2.imread("/data/Y.Disk/work/urinalysis/images/stripe experiments/compact/template_mirror_no_stroke_border.png", cv2.IMREAD_GRAYSCALE), 1)

	cap = cv2.VideoCapture(0)
	while cap.isOpened():
		_, frame = cap.read()
		scene_image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
		matched_image = match_image(scene_image.copy(), query_image.copy())
		cv2.imshow("match", matched_image)
		cv2.waitKey(1)

if __name__ == "__main__":
	main()