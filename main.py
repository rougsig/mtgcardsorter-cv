import numpy as np
import imutils
import matplotlib.pyplot as plt
import cv2 as cv
import pickle


def order_points(pts):
  # initialzie a list of coordinates that will be ordered
  # such that the first entry in the list is the top-left,
  # the second entry is the top-right, the third is the
  # bottom-right, and the fourth is the bottom-left
  rect = np.zeros((4, 2), dtype="float32")
  # the top-left point will have the smallest sum, whereas
  # the bottom-right point will have the largest sum
  s = pts.sum(axis=1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
  # now, compute the difference between the points, the
  # top-right point will have the smallest difference,
  # whereas the bottom-left will have the largest difference
  diff = np.diff(pts, axis=1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  # return the ordered coordinates
  return rect


def four_point_transform(image, pts):
  # obtain a consistent order of the points and unpack them
  # individually
  rect = order_points(pts)
  (tl, tr, br, bl) = rect
  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")
  # compute the perspective transform matrix and then apply it
  M = cv.getPerspectiveTransform(rect, dst)
  warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
  # return the warped image
  return warped


if __name__ == '__main__':
  # load the image and compute the ratio of the old height
  # to the new height, clone it, and resize it
  image = cv.imread("456.jpg")
  ratio = image.shape[0] / 500.0
  orig = image.copy()
  image = imutils.resize(image, height=500)
  # convert the image to grayscale, blur it, and find edges
  # in the image
  # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  # gray = cv.GaussianBlur(gray, (5, 5), 0)
  # edged = cv.Canny(gray, 75, 200)
  # # show the original image and the edge detected image
  # print("STEP 1: Edge Detection")
  # # cv.imshow("Image", image)
  # # cv.imshow("Edged", edged)
  # # cv.waitKey(0)
  # # cv.destroyAllWindows()
  #
  # # find the contours in the edged image, keeping only the
  # # largest ones, and initialize the screen contour
  # cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
  # cnts = imutils.grab_contours(cnts)
  # cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]
  # # loop over the contours
  # for c in cnts:
  #   # approximate the contour
  #   peri = cv.arcLength(c, True)
  #   approx = cv.approxPolyDP(c, 0.02 * peri, True)
  #   # if our approximated contour has four points, then we
  #   # can assume that we have found our screen
  #   if len(approx) == 4:
  #     screenCnt = approx
  #     break
  # # show the contour (outline) of the piece of paper
  # print("STEP 2: Find contours of paper")
  # cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
  # # cv.imshow("Outline", image)
  # # cv.waitKey(0)
  # # cv.destroyAllWindows()
  #
  # # apply the four point transform to obtain a top-down
  # # view of the original image
  # warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
  # # convert the warped image to grayscale, then threshold it
  # # to give it that 'black and white' paper effect
  # warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
  # # show the original and scanned images
  # print("STEP 3: Apply perspective transform")
  # # cv.imshow("Original", imutils.resize(orig, height=650))
  # # cv.imshow("Scanned", imutils.resize(warped, height=650))
  # # cv.waitKey(0)

  img1 = cv.imread('123.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
  img2 = cv.imread('456.jpg', cv.IMREAD_GRAYSCALE)
  # Initiate SIFT detector
  sift = cv.SIFT_create()
  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)
  # FLANN parameters
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)  # or pass empty dictionary
  flann = cv.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des1, des2, k=2)
  # Need to draw only good matches, so create a mask
  matchesMask = [[0, 0] for i in range(len(matches))]
  good_points = []
  # ratio test as per Lowe's paper
  for i, (m, n) in enumerate(matches):
    if m.distance < 0.6 * n.distance:
      good_points.append(m)
      matchesMask[i] = [1, 0]
  draw_params = dict(matchColor=(0, 255, 0),
                     singlePointColor=(255, 0, 0),
                     matchesMask=matchesMask,
                     flags=cv.DrawMatchesFlags_DEFAULT)

  number_keypoints = 0
  if len(kp1) <= len(kp2):
    number_keypoints = len(kp1)
  else:
    number_keypoints = len(kp2)
  print("Keypoints 1ST Image: " + str(len(kp1)))
  print("Keypoints 2ND Image: " + str(len(kp2)))

  print("GOOD Matches:", len(good_points))
  print("How good it's the match: ", len(good_points) / number_keypoints * 100, "%")

  img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
  plt.imshow(img3, ), plt.show()

  with open('fileNameToSave.txt', 'wb') as f:
    pickle.dump(des1, f, -1)  # -1 for best compression available
