import numpy as np
from cv2 import data
from PyQt5.QtGui import QPixmap
from  PyQt5.QtWidgets import  QMainWindow, QApplication,QLabel,QPushButton
from PyQt5 import uic
import sys
import cv2
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        uic.loadUi('line.ui',self)
        self.label = self.findChild(QLabel,"label")
        self.image_label = self.findChild(QLabel,"image1")
        self.line_image_label = self.findChild(QLabel, "line")
        self.camera_btn = self.findChild(QPushButton, "movie")
        self.image_btn = self.findChild(QPushButton, "image")
        self.show()

        self.image_btn.clicked.connect(self.lane_in_image)
        self.camera_btn.clicked.connect(self.lane_in_video)



    def lane_in_image(self):
        def  canny_edge(image):
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_img = cv2.dilate(gray_img, kernel=np.ones((5, 5), np.uint8))

            # Step 3) Canny
            canny = cv2.Canny(gray_img, 100, 200)
            return canny
        def ROI_mask(image):
            roi_vertices = [(270, 650), (700, 390), (1100, 720)]
            triangle =np.array([roi_vertices], np.int32)
            mask = np.zeros_like(image)
            # 255 is mask color
            cv2.fillPoly(mask, triangle, 255)
            masked_img = cv2.bitwise_and(image, mask)
            return masked_img

        def draw_lines(image, lines):
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 120), 2)

            # combined_image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
            return image

        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '', "Image files (*.jpg *.png)")

        self.pixmap_size = QSize(self.image_label.width(), self.image_label.height())
        self.pixmap = QPixmap(fname[0])
        self.pixmap = self.pixmap.scaled(self.pixmap_size)
        self.image_label.setPixmap(self.pixmap)
        img = cv2.imread(fname[0])
        canny_edges = canny_edge(img)
        # cv2.imshow('img', canny_edges)
        cropped_image = ROI_mask(canny_edges)
        # cv2.imshow('img1', cropped_image)
        lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi / 180, threshold=85,
                                lines=np.array([]), minLineLength=100, maxLineGap=20)
        # average_lines = avg_lines(img, lines)
        combind_image = draw_lines(img, lines)
        haar_cascade = 'cars.xml'
        car_cascade = cv2.CascadeClassifier(haar_cascade)
        # convert frames to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detects cars of different sizes in the input image
        cars = car_cascade.detectMultiScale(gray, 1.6, 3)
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.imwrite('line.jpg', img)
        self.pixmap_detect_size = QSize(self.line_image_label.width(), self.line_image_label.height())
        print(self.pixmap_detect_size)
        self.pixmap_detect = QPixmap('line.jpg')
        print(self.pixmap_detect)
        self.pixmap_detect = self.pixmap_detect.scaled(self.pixmap_detect_size)
        self.line_image_label.setPixmap(self.pixmap_detect)
    def lane_in_video(self):
        def perspective(inputImage):
            img_size = (inputImage.shape[1], inputImage.shape[0])
            pers_point = np.float32([[590, 440],
                                                     [690, 440],
                                                     [200, 640],
                                                     [1000, 640]])
            dst = np.float32([[200, 0],
                              [1200, 0],
                              [200, 710],
                              [1200, 710]])
            matrix_wrap = cv2.getPerspectiveTransform(pers_point, dst)
            minv = cv2.getPerspectiveTransform(dst, pers_point)
            b_eyes = cv2.warpPerspective(inputImage, matrix_wrap, img_size)

            return b_eyes,minv
        def  process_img(inputImage):
            lower_white = np.array([0, 160, 10])
            upper_white = np.array([255, 255, 255])

            mask = cv2.inRange(inputImage, lower_white, upper_white)
            hls_result = cv2.bitwise_and(inputImage, inputImage, mask= mask)
            gray = cv2.cvtColor(hls_result, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
            return thresh

        def  plotHistogrom(thresh):
            histogrom  = np.sum(thresh[thresh.shape[0] // 2:, :], axis= 0)
            return histogrom

        def slide_window_search(binary_warped, histogram):

            # Find the start of left and right lane lines using histogram info
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            midpoint = np.int64(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            # A total of 9 windows will be used
            nwindows = 9
            window_height = np.int64(binary_warped.shape[0] / nwindows)
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            leftx_current = leftx_base
            rightx_current = rightx_base
            margin = 100
            minpix = 50
            left_lane_inds = []
            right_lane_inds = []

            for window in range(nwindows):
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                              (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                              (0, 255, 0), 2)
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                if len(good_left_inds) > minpix:
                    leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))
            #### END - Loop to iterate through windows and search for lane lines #######

            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Apply 2nd degree polynomial fit to fit curves
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            return  left_fit, right_fit

        def general_search(binary_warped, left_fit, right_fit):

            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 100
            left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                           left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                                 left_fit[1] * nonzeroy + left_fit[
                                                                                     2] + margin)))

            right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                            right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                                   right_fit[1] * nonzeroy + right_fit[
                                                                                       2] + margin)))

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            ## VISUALIZATION ###########################################################

            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            window_img = np.zeros_like(out_img)
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                            ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

            ret = {}
            ret['leftx'] = leftx
            ret['rightx'] = rightx
            ret['left_fitx'] = left_fitx
            ret['right_fitx'] = right_fitx
            ret['ploty'] = ploty

            return ret



        def draw_lane_lines(original_image, warped_image, Minv, draw_info):
            left_fitx = draw_info['left_fitx']
            right_fitx = draw_info['right_fitx']
            ploty = draw_info['ploty']

            warp_zero = np.zeros_like(warped_image).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            mean_x = np.mean((left_fitx, right_fitx), axis=0)
            pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 255, 255))

            newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
            result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

            return  result





        ym_per_pix = 30 / 720
        # Standard lane width is 3.7 meters divided by lane width in pixels which is
        # calculated to be approximately 720 pixels not to be confused with frame height
        xm_per_pix = 3.7 / 720
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '', "Video Files  (*.mp4 )")
        video = cv2.VideoCapture(fname[0])

        while True:
            _,frame = video.read()
            birdView, minverse = perspective(frame)

            thresh= process_img(birdView)

            hist= plotHistogrom(thresh)

            left_fit,right_fit = slide_window_search(thresh, hist)
            draw_info = general_search(thresh, left_fit, right_fit)
            result = draw_lane_lines(frame, thresh, minverse, draw_info)


            # Displaying final image


            haar_cascade = 'cars.xml'
            car_cascade = cv2.CascadeClassifier(haar_cascade)
            # convert frames to gray scale
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            # Detects cars of different sizes in the input image
            cars = car_cascade.detectMultiScale(result, 1.3, 2)
            for (x, y, w, h) in cars:
                cv2.rectangle(result,  (x, y), (x + w, y + h), (51, 51, 255), 2)
            cv2.imshow("Final", result)
            # Wait for the ENTER key to be pressed to stop playback
            if cv2.waitKey(1) == 13:
                break

    # Cleanup
        video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    app=QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()

