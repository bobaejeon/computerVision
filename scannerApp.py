import numpy as np
import cv2 as cv

# 1. edge detection(GaussianBlur+Canny) 2. finding contour
# 3. finding vertices coordinates 4. perspective transform

# 1. edge detection
src = cv.imread("squarePaper.jpg")
if src is None:
    print('Image load failed')
    exit()

src = cv.resize(src, (0,0), fx=0.6, fy=0.6, interpolation=cv.INTER_NEAREST)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

src_gray = cv.GaussianBlur(src_gray, (3, 3), 0)  # to remove noise

edge = cv.Canny(src_gray, 100, 200)  # to find edges

# 2. finding contour https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html
contours, _ = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key=cv.contourArea, reverse=True) # sort contours from the largest area(desc. order)

# 3. finding vertices coordinates
for i in range(len(contours)):
    # contour approximation https://docs.opencv.org/3.4/db/d00/samples_2cpp_2squares_8cpp-example.html#a20
    approx = cv.approxPolyDP(contours[i], cv.arcLength(contours[i], True) * 0.02, True)
    # if the polygon has 4 vertices, that can be considered as a rectangle
    if len(approx) == 4:
        break

approx = approx.reshape(len(approx), np.size(approx[0]))

xSubY = np.subtract(approx[:, 0], approx[:, 1])
xAddY = approx.sum(axis=1)

src_pts = np.zeros((4, 2), dtype=np.float32)
src_pts[0, :] = approx[np.where(xAddY == np.min(xAddY))].reshape(2)  # min(x+y)
src_pts[1, :] = approx[np.where(xSubY == np.max(xSubY))].reshape(2)  # max(x-y)
src_pts[2, :] = approx[np.where(xAddY == np.max(xAddY))].reshape(2)  # max(x+y)
src_pts[3, :] = approx[np.where(xSubY == np.min(xSubY))].reshape(2)  # min(x-y)

# 4. perspective transform
w = int(np.max(src_pts,axis=0)[0] - np.min(src_pts,axis=0)[0])
h = int(np.max(src_pts,axis=0)[1] - np.min(src_pts,axis=0)[1])

dst_pts = np.array([[0,0],
                    [w-1,0],
                    [w-1,h-1],
                    [0,h-1]]).astype(np.float32)

pers_mat = cv.getPerspectiveTransform(src_pts,dst_pts)
dst = cv.warpPerspective(src, pers_mat,(w,h))

cv.imshow('src', src)
cv.imshow('dst', dst)
cv.waitKey()
cv.destroyAllWindows()