import cv2 as cv

global src, src_hsv


def on_hue_changed(_=None):
    lower_hue = cv.getTrackbarPos('Lower', 'Color Range')
    upper_hue = cv.getTrackbarPos('Upper', 'Color Range')
    lowerb = (lower_hue, 100, 0)  # hsv
    upperb = (upper_hue, 255, 255)
    mask = cv.inRange(src_hsv, lowerb, upperb)

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    dst = cv.bitwise_and(src, mask)

    cv.imshow('Color Range', dst)


def main():
    global src, src_hsv

    src = cv.imread('candies.jpg', cv.IMREAD_COLOR)
    src = cv.resize(src, (0, 0), fx=0.6, fy=0.6, interpolation=cv.INTER_NEAREST)
    src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

    cv.imshow('src', src)

    cv.namedWindow('Color Range')
    cv.createTrackbar('Lower', 'Color Range', 40, 179, on_hue_changed)
    cv.createTrackbar('Upper', 'Color Range', 80, 179, on_hue_changed)
    on_hue_changed(0)

    cv.waitKey()
    cv.destroyAllWindows()


main()
