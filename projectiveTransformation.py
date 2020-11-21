import cv2 as cv
import numpy as np


def on_mouse(event, x, y, flags, param):
    global cnt, src_pts
    if event == cv.EVENT_LBUTTONDOWN:
        if cnt < 4:
            src_pts[cnt, :] = np.array([x, y]).astype(np.float32)
            cnt += 1
            cv.circle(src, (x, y), 5, (0, 0, 255), -1)
            cv.imshow('src', src)
        elif cnt == 4:
            w = int(np.amax(src_pts, axis=0)[0] - np.amin(src_pts, axis=0)[0])
            h = int(np.amax(src_pts, axis=0)[1] - np.amin(src_pts, axis=0)[1])

            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            pers_mat = cv.getPerspectiveTransform(src_pts, dst_pts)
            dst = cv.warpPerspective(src, pers_mat, (w, h))
            cv.imshow('dst', dst)


cnt = 0
src_pts = np.zeros([4, 2], dtype=np.float32)
src = cv.imread('card.bmp')

cv.namedWindow('src')
cv.setMouseCallback('src', on_mouse)
cv.imshow('src', src)
cv.waitKey()
cv.destroyAllWindows()
