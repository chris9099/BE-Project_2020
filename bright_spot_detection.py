import cv2
import imutils

def find_bright_spots(image):
	'''
		retuns bright spots in the center of the frame.
	'''

	# get the image dimensions
	height, width, _ = image.shape
	# value by which the center of the image should be added an subtracted to get a window for detection
	cropping_threshold = 200
	# area above which bright spots should be cansidered
	threshold_area = 5000	

	# cropping the image, take the entire height but width using cropping_threshold
	image = image[0:height, width // 2 - cropping_threshold:width // 2 + cropping_threshold]

	# convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blur the image
	blurred = cv2.GaussianBlur(gray, (11, 11), 0)

	# replace all pixels with value more than 170 by 255 and others to 0 to get all the bright ares
	thresh = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)[1]

	# remove noise from the threshold image
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=4)

	# find contours, areas with same intensity
	cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)

	# grab all the contours
	cnts = imutils.grab_contours(cnts)

	# find all the contours with area greater than the threshold_area
	cnts = list(filter(lambda cnt:cv2.contourArea(cnt) > threshold_area, cnts))
	# print("*" * 10)
	# for cnt in cnts:
	# 	print(cv2.contourArea(cnt))

	# print("*" * 10)
	return len(cnts)
