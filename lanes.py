# from sklearn.cluster import KMeans
import numpy as np
# import matplotlib.pyplot as plt
import cv2
import os
from dominantcolors import get_dominant_colors_for


class LaneMarker:

    def __init__(self, wdir=str(os.getcwd)):
        self.camnum = 1
        self.cap = cv2.VideoCapture(self.camnum)
        self.wdir = wdir
        # self.img_source_path = os.path.join(self.wdir, "test_images/vid/1440.jpg")
        self.img_source_path = os.path.join(self.wdir, "Users\administrator\Documents\School\Senior_Design\vid\1300.jpg")
        self.img_hist_ref_path = os.path.join(self.wdir, "Users\administrator\Documents\School\Senior_Design\vid\0000.jpg")
       

        # self.hsv_lb = {"h" : 0, "s" : 28, "v" : 153}
        # self.hsv_ub = {"h" : 41, "s" : 89, "v" : 255}
        self.hsv_lb = {"h" : 0, "s" : 15, "v" : 142}
        self.hsv_ub = {"h" : 179, "s" : 100, "v" : 255}
        # self.hsv_lb = {"h" : 146, "s" : 84, "v" : 104}    # YUV
        # self.hsv_ub = {"h" : 221, "s" : 123, "v" : 148}   # YUV

        self.img_source = self.resize_image( cv2.imread(self.img_source_path) )
        self.img_hist_ref = self.resize_image( cv2.imread(self.img_hist_ref_path) )
        self.img_processed = self.process_image( self.img_source )
    def set_video_source(self, camnum):
        self.camnum = camnum
        self.cap = cv2.VideoCapture(camnum)
    def set_image_source(self, img):
        self.img_source = img
        self.img_processed = self.process_image( img )

    def set_image_source_rel_path(self, path):
        self.img_source_path = os.path.join(self.wdir, path)
        self.set_image_source( self.resize_image( cv2.imread(self.img_source_path) ) )


    def set_hsv_lb(self, component, value):
        self.hsv_lb[component] = value


    def set_hsv_ub(self, component, value):
        self.hsv_ub[component] = value


    def hsv_mask(self, input):
        hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        hue = (self.hsv_lb["h"], self.hsv_ub["h"])
        sat = (self.hsv_lb["s"], self.hsv_ub["s"])
        val = (self.hsv_lb["v"], self.hsv_ub["v"])
        mask = cv2.inRange(hsv, (hue[0], sat[0], val[0]), (hue[1], sat[1], val[1]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = self.compute_gradient(mask)
        return cv2.bitwise_and(input, input, mask=mask), mask


    def lab_mask(self, input):
        lab = cv2.cvtColor(self.img_source, cv2.COLOR_BGR2LAB)
        # lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        hue = (self.hsv_lb["h"], self.hsv_ub["h"])
        sat = (self.hsv_lb["s"], self.hsv_ub["s"])
        val = (self.hsv_lb["v"], self.hsv_ub["v"])
        mask = cv2.inRange(lab, (hue[0], sat[0], val[0]), (hue[1], sat[1], val[1]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.bitwise_and(input, input, mask=mask), mask


    def resize_image(self, img, w=320, h=240):
        return cv2.resize(img, (w, h))
        # return img
        

    def hist_match(self, source, template):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Author: ali_m  via StackExchange

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)


    def hist_match_3ch(self, source, target):
        img_matched = np.zeros(source.shape, dtype=source.dtype)
        for i in range(img_matched.shape[2]):
            img_matched[:,:,i] = self.hist_match(
                source[:,:,i],
                target[:,:,i]
            )
        return img_matched


    def quantize(self, img, k=2, method="kmeans"):
        # _img = img
        _img = self.resize_image(img, 160, 120)
        if method == "kmeans":
            Z = _img.reshape((-1,3))
            # convert to np.float32
            Z = np.float32(Z)
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS  + cv2.TERM_CRITERIA_MAX_ITER, 20, 1)
            K = k
            _,label,center=cv2.kmeans(Z,K,None,criteria,1,cv2.KMEANS_PP_CENTERS)
            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((_img.shape))
        elif method == "constant_means":
            L2norm = lambda x: np.sqrt(np.sum(np.abs(x)**2, axis=-1))
            print("L2norm", L2norm)
            fixed_center = np.array([
                [97.27209473, 54.49308014, 117.63933563],
                [34.77519226, 144.03575134, 127.25043488]
            ], dtype=np.int16)
            H1, S1, V1 = fixed_center[0]
            H2, S2, V2 = fixed_center[1]
            h_ch = np.zeros(_img.shape[:-1], dtype=np.int16)
            s_ch = np.zeros(_img.shape[:-1], dtype=np.int16)
            v_ch = np.zeros(_img.shape[:-1], dtype=np.int16)
            _img_int16 = _img.astype(np.int16)
            dist_to_center0 = np.array( [L2norm(v - fixed_center[0]) for v in _img_int16] )
            dist_to_center1 = np.array( [L2norm(v - fixed_center[1]) for v in _img_int16] )
            h_ch[:,:] = H1; h_ch[dist_to_center0 > dist_to_center1] = H2
            s_ch[:,:] = S1; s_ch[dist_to_center0 > dist_to_center1] = S2
            v_ch[:,:] = V1; v_ch[dist_to_center0 > dist_to_center1] = V2
            res2 = np.zeros(_img.shape, dtype=np.uint8)
            res2[:,:,0] = h_ch
            res2[:,:,1] = s_ch
            res2[:,:,2] = v_ch       
        else:
            _, res2 = get_dominant_colors_for(_img, k)

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


    def hsv_trackbars(self, wname='Color Space Thresholding'):
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('H_lb', wname, 0, 255, lambda x: self.set_hsv_lb("h", x))
        cv2.createTrackbar('H_ub', wname, 0, 255, lambda x: self.set_hsv_ub("h", x))
        cv2.createTrackbar('S_lb', wname, 0, 255, lambda x: self.set_hsv_lb("s", x))
        cv2.createTrackbar('S_ub', wname, 0, 255, lambda x: self.set_hsv_ub("s", x))
        cv2.createTrackbar('V_lb', wname, 0, 255, lambda x: self.set_hsv_lb("v", x))
        cv2.createTrackbar('V_ub', wname, 0, 255, lambda x: self.set_hsv_ub("v", x))
        cv2.setTrackbarPos('H_lb', wname, self.hsv_lb["h"])
        cv2.setTrackbarPos('H_ub', wname, self.hsv_ub["h"])
        cv2.setTrackbarPos('S_lb', wname, self.hsv_lb["s"])
        cv2.setTrackbarPos('S_ub', wname, self.hsv_ub["s"])
        cv2.setTrackbarPos('V_lb', wname, self.hsv_lb["v"])
        cv2.setTrackbarPos('V_ub', wname, self.hsv_ub["v"])
        while True:
            #self.img_processed = self.process_image( self.img_source )
            self.img_processed = self.process_image_cam()
            self.show_image(img=self.img_processed, wname=wname)
            k = cv2.waitKey(50) & 0xFF
            if k == ord('q'):
                break
        cv2.destroyAllWindows()
        

    def process_image(self, img):
        out = img
        out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        out = self.adjust_gamma(out, 1.25)
        # out = self.hist_match_3ch(out, self.img_hist_ref)
        out = self.add_blur(out)
        out = self.quantize(out, 2, "constant_means")
        # out, _ = self.hsv_mask(out)
        # out, _ = self.lab_mask(out)
        return out

    def process_image_cam(self):
        ret, out = self.cap.read()
        out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        out = self.adjust_gamma(out, 1.25)
        # out = self.hist_match_3ch(out, self.img_hist_ref)
        out = self.add_blur(out)
        out = self.quantize(out, 2, "constant_means")
        # out, _ = self.hsv_mask(out)
        # out, _ = self.lab_mask(out)
        return out


    def write_on_image(self, img, text, text_origin=None):
        if text_origin is None:
            text_origin = (10, img.shape[0]-10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 0, 0)
        font_scale = 1
        font_thickness = 2
        return cv2.putText( img, text, text_origin, 
                font, font_scale, font_color, font_thickness, cv2.LINE_AA )


    def run_video_frames(self):
        fileID = 1000
        run = True
        while fileID <= 9414:
            while run:
                rel_path = 'test_images/vid/{0:04d}.jpg'.format(fileID)
                self.set_image_source_rel_path(rel_path)
                display_image = self.write_on_image(self.img_source, '{0:04d}.jpg'.format(fileID))
                self.show_image(img=self.skeleton(self.img_processed), wname="skeleton")
                self.show_image(img=(display_image, self.img_processed), wname="quantized")
                # self.show_image( 
                #     img=(
                #         display_image, 
                #         self.quantize(self.img_processed, 2),
                #         get_dominant_colors_for(self.img_processed, 4)[1]
                #     ) 
                # )

                k = cv2.waitKey(5) & 0xFF
                if k == ord('q'):
                    run = False
                    break
                if k == ord('p'):
                    run = not run
                fileID += 1
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') and run is False:
                    break
            if k == ord('p'):
                run = not run
        cv2.destroyAllWindows()


obj = LaneMarker()
# obj.show_image_block(img=(obj.img_source, obj.quantize(cv2.cvtColor(obj.img_source, cv2.COLOR_BGR2HSV),2)))
# obj.show_image_block(img=obj.compute_gradient(obj.img_processed))
# obj.show_image_block(img=(obj.img_source, cv2.cvtColor(obj.img_source, cv2.COLOR_BGR2HSV)))
# obj.show_image_block(img=obj.compute_gradient(obj.hsv_mask(obj.img_source)[1]))
# kernel = np.ones((3,3),np.uint8)


# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# test = obj.quantize(cv2.cvtColor(obj.img_source, cv2.COLOR_BGR2HSV),2)
# obj.show_separate_channels(test)

# obj.show_image_block(
#     # img=obj.compute_gradient(
#     #     cv2.morphologyEx(test[:,:,1], cv2.MORPH_OPEN, kernel)
#     # )
#     img=cv2.morphologyEx(test[:,:,1], cv2.MORPH_OPEN, kernel)
# )

obj.run_video_frames()
