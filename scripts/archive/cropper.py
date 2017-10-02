import cv2
import glob
import numpy as np
from os.path import join
import sys
sys.path.append("table/")
from table import Table

source_dir = "/data/Y.Disk/work/urinalysis/images/Photos_22_09_17/XiaomiD_sysmex/"
verts = []
image = None
image_copy = None
image_crop = None
verts_query = [(0,0), (625, 0), (625, 300), (0, 300)]

def get_correct_arrangement(rect):
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

def get_warp_matrix(verts, verts_query):
	src_pts = np.expand_dims(np.array(verts)[get_correct_arrangement(verts)], 1)
	dst_pts = np.expand_dims(np.array(verts_query), 1)
	return cv2.findHomography(src_pts, dst_pts)[0]

def downsample(image, to_width=960):
	to_height = int(image.shape[0] * to_width / image.shape[1])
	return cv2.resize(image, (to_width, to_height))

def click_and_set_point(event, x, y, flags, param):
	global verts, image_copy
	if event == cv2.EVENT_LBUTTONDBLCLK:
		verts.append((x, y))
		cv2.circle(image_copy, (x, y), image_copy.shape[0] / 150, (0, 255, 0),  -1)

def main():
	global verts, verts_query, image, image_copy, image_crop
	images_names = glob.glob(join(source_dir, "*.*"))
	print "Found {} images".format(len(images_names))
	for j, image_name in enumerate(images_names, 1):
		print "({} / {}) Reading image {}".format(j, len(images_names), image_name)
		image = downsample(cv2.imread(image_name, cv2.IMREAD_COLOR))
		image_copy = image.copy()
		image_crop = np.zeros(shape=verts_query[2][::-1])
		cv2.namedWindow("Image")
		cv2.setMouseCallback("Image", click_and_set_point)
		while True:
			cv2.imshow("Crop", image_crop)
			cv2.imshow("Image", image_copy)

			key = cv2.waitKey(10)
			if key == ord('r'):
				verts = []
				image_copy = image.copy()
			if len(verts) == 4:
				hull = cv2.convexHull(np.array(verts).reshape(-1, 1, 2))
				cv2.drawContours(image_copy, [hull], 0, (0, 255, 0), image_copy.shape[0] / 250)

				warp_matrix = get_warp_matrix(verts, verts_query)
				image_crop = cv2.warpPerspective(src=image, M=warp_matrix, dsize=verts_query[2])
			if key == ord('s'):
				new_name = image_name.replace(".jpg", "-cropped.jpg")
				cv2.imwrite(new_name, image_crop)
				print "Saved {}".format(new_name)
			if key == ord('n'):
				verts = []
				break

if __name__ == "__main__":
	main()