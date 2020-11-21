import numpy as np
import cv2 as cv

# Goal: to find every "large" square and to projective-transform it

# 1. edge detection
src = cv.imread('cards.jpg')
if src is None:
    print('Image load failed')
    exit()

src = cv.resize(src, (0, 0), fx=0.6, fy=0.6, interpolation=cv.INTER_NEAREST)
cv.imshow('src', src)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

src_gray = cv.GaussianBlur(src_gray, (3, 3), 0)  # to remove noise
#src_gray = np.float32(src_gray)

edge = cv.Canny(src_gray, 150, 250)  # to find edges

# 2. finding contour https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html
contours, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # we don't need contours in contours
contours = sorted(contours,key=cv.contourArea, reverse=True)

# 3. finding vertices coordinates
added = []
img_size = src.shape[0] * src.shape[1]
max_size = int(img_size * 0.3)
min_size = int(img_size * 0.05)
for i in range(len(contours)):
    # contour approximation https://docs.opencv.org/3.4/db/d00/samples_2cpp_2squares_8cpp-example.html#a20
    approx = cv.approxPolyDP(contours[i], cv.arcLength(contours[i], True)*0.01, True)
    # if the polygon has 4 vertices, that can be considered as a rectangle
    if len(approx) == 4 and max_size > cv.contourArea(approx) > min_size:
        added.append(approx.reshape(4, 2))

# show 4 vertices
dst = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)

i = 0
for vertices in added:
    i += 1
    xSubY = np.subtract(vertices[:, 0], vertices[:, 1])
    xAddY = vertices.sum(axis=1)

    src_pts = np.zeros((4, 2), dtype=np.float32)
    src_pts[0, :] = vertices[np.where(xAddY == np.min(xAddY))].reshape(2)  # min(x+y)
    src_pts[1, :] = vertices[np.where(xSubY == np.max(xSubY))].reshape(2)  # max(x-y)
    src_pts[2, :] = vertices[np.where(xAddY == np.max(xAddY))].reshape(2)  # max(x+y)
    src_pts[3, :] = vertices[np.where(xSubY == np.min(xSubY))].reshape(2)  # min(x-y)

    # 4. perspective transform
    w = int(max(abs(src_pts[1][0] - src_pts[0][0]), abs(src_pts[2][0] - src_pts[3][0])))
    h = int(max(abs(src_pts[1][1] - src_pts[2][1]), abs(src_pts[0][1] - src_pts[3][1])))

    dst_pts = np.array([[0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1],
                        [0, h - 1]]).astype(np.float32)

    pers_mat = cv.getPerspectiveTransform(src_pts, dst_pts)
    dst = cv.warpPerspective(src, pers_mat, (w, h))

    cv.imshow('dst'+str(i), dst)
    cv.waitKey()
    cv.destroyWindow('dst'+str(i))

cv.waitKey()
cv.destroyAllWindows()
