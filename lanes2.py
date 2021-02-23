# from sklearn.cluster import KMeans
import numpy as np
# import matplotlib.pyplot as plt
import cv2
import os


class LaneMarker:

    def __init__(self, cap):
        # self.capture = cv2.VideoCapture(0)
        self.capture = cap
        self.img_source = self.resize_image( self.capture.read()[1])
        self.img_processed = self.process_image( self.img_source )


    def resize_image(self, img, w=320, h=240):
        return cv2.resize(img, (w, h))
        # return img

    def quantize(self, img, k=2):
        _img = self.resize_image(img, 160, 120)
        Z = _img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS  + cv2.TERM_CRITERIA_MAX_ITER, 20, 1)
        K = k
        _, label , center = cv2.kmeans(Z, K, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((_img.shape))

        return self.resize_image(res2)


    def compute_gradient(self, img):
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # grad = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
        grad = cv2.Laplacian(img, cv2.CV_64F)
        return grad.astype(img.dtype)


    def skeleton(self, img):
        img = img[:,:,0]
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        colors = np.unique(img)
        _, img = cv2.threshold(img, colors[0], colors[-1], cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # img = cv2.adaptiveThreshold(
            # img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            # cv2.THRESH_BINARY, 11, 7
        # )
        # return img
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        done = False
        while(not done):
            eroded = cv2.erode(img, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True
        return skel


    def add_blur(self, input, val=7):
        is_odd = lambda x: x % 2 == 1
        if is_odd(val):
            return cv2.medianBlur(input, val)
        else:
            return cv2.medianBlur(input, val+1)
        # return cv2.GaussianBlur(input, (13,13), 2.0, 2.0)
    

    def adjust_brightness(self, input, beta):
        beta = np.floor(255.0 / 100.0 * beta)
        return cv2.convertScaleAbs(input, alpha=1.0, beta=beta)


    def adjust_contrast(self, input, alpha):
        alpha = alpha / 50.0 + 1
        return cv2.convertScaleAbs(input, alpha=alpha, beta=0)
    

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)


    def show_image(self, img=None, wname=None):
        if wname is None:
            wname = 'image'
        if img is None:
            img = self.img_processed
        if type(img) is tuple:
            img = np.concatenate(img, axis=1)
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.imshow(wname, img)


    def show_image_block(self, img=None, wname=None):
        if wname is None:
            wname = 'image'
        if img is None:
            img = self.img_processed
        if type(img) is tuple:
            img = np.concatenate(img, axis=1)
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.imshow(wname, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def show_separate_channels(self, image):
        b = image.copy()
        # set green and red channels to 0
        b[:, :, 1] = 0
        b[:, :, 2] = 0


        g = image.copy()
        # set blue and red channels to 0
        g[:, :, 0] = 0
        g[:, :, 2] = 0

        r = image.copy()
        # set blue and green channels to 0
        r[:, :, 0] = 0
        r[:, :, 1] = 0

        self.show_image_block(img=(b,g,r))
    def  birdsEyeTrans(self):
        IMAGE_H = 480
        IMAGE_W =  640

        src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
        Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation
        img = self.process_image(self, self.capture)  # Read the test img
        img = img[450:(450 + IMAGE_H), 0:IMAGE_W]  # Apply np slicing for ROI crop
        warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))  # Image warping
        cv2.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))  # Show results
        cv2.show()

    def process_image(self, img):
        out = img
        out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        out = self.adjust_gamma(out, 1.25)
        out = self.add_blur(out)
        out = self.quantize(out, 2)
        return out


    def run_video_frames(self):
        run = True
        while True:
            while run:
                self.img_source = self.resize_image( self.capture.read()[1] )
                self.img_processed = self.process_image( self.img_source )
                self.show_image()
                k = cv2.waitKey(5) & 0xFF
                if k == ord('q'):
                    run = False
                    break
                if k == ord('p'):
                    run = not run
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') and run is False:
                    break
            if k == ord('p'):
                run = not run
        cv2.destroyAllWindows()


#obj = LaneMarker()
#obj.run_video_frames()
# obj.birdsEyeTrans()

