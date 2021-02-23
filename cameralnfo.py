import cv2
import numpy as np

'''
A class that is in charge of flattening an image from a camera given
a known height of the camera
A known distance to the closest thing the camera can see
the width that is seen at that closest distance
a known distance to the "mioddle" of the screen (halfway up what is seen)
a knowd width that is seen at that middle
It then be wizard
'''


class CameraInfo:
    def __init__(self, h, d1, w1, d2, w2,fx,fy):
        self.height = h
        self.closedist = d1
        self.closewidth = w1
        self.middist = d2
        self.midwidth = w2
        self.per_inch = 1
        self.map_height = fy
        self.map_width = fx
        self.trim = int(((float(self.midwidth - self.closewidth)/2)/float(self.middist - self.closedist))*self.map_height)
        self.rectangle_height = int(.88*self.map_height)#number is ratio for how far down the frame rectangle should start

    def convertToFlat(self, image):
        pts_src = np.float32([[0, int(self.map_height)], [self.map_width, int(self.map_height)], [0, 0], [self.map_width, 0]])
        left_trim = self.trim
        right_trim = self.map_width - self.trim
        pts_dst = np.float32([[left_trim, self.map_height],[right_trim, self.map_height],[0, 0],[self.map_width,0]])
        mtx = cv2.getPerspectiveTransform(pts_src,pts_dst)
        flat_map = cv2.warpPerspective(image, mtx, (self.map_width, self.map_height))
        cv2.rectangle(flat_map, (60+left_trim,self.rectangle_height), (right_trim-60,self.map_height), (0,0,0), -1)#occlude bender chassis
        return flat_map

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def quantize(self, img, k=2):
        _img = self.resize_image(img, 160, 120)
        Z = _img.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1)
        K = k
        _, label, center = cv2.kmeans(Z, K, None, criteria, 1, cv2.KMEANS_PP_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((_img.shape))

    def resize_image(self, img, w=320, h=240):
        return cv2.resize(img, (w, h))

    def add_blur(self, input, val=7):
        is_odd = lambda x: x % 2 == 1
        if is_odd(val):
            return cv2.medianBlur(input, val)
        else:
            return cv2.medianBlur(input, val + 1)
    def process_image(self, img):
        out = img
      #  out = cv2.cvtColor(np.float32(out), cv2.COLOR_RGB2GRAY)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        out = self.adjust_gamma(out, 1.25)
        out = self.add_blur(out)
        out = self.quantize(out, 2)
        return out
