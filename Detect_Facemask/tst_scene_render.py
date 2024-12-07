#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
from numpy import pi, sin, cos
import cv2 as cv

defaultSize = 512

class TestSceneRender:
    def __init__(self, bgImg=None, fgImg=None, deformation=False, speed=0.25, **params):
        self.time = 0.0
        self.timeStep = 1.0 / 30.0
        self.deformation = deformation
        self.speed = speed

        # Sử dụng ảnh nền nếu có, nếu không tạo ảnh đen mặc định
        if bgImg is not None:
            self.sceneBg = bgImg.copy()
        else:
            self.sceneBg = np.zeros((defaultSize, defaultSize, 3), np.uint8)

        # Đặt kích thước ảnh nền
        self.h, self.w = self.sceneBg.shape[:2]

        # Nếu có ảnh tiền cảnh, thay đổi kích thước để phù hợp với ảnh nền và thiết lập vị trí ban đầu
        if fgImg is not None:
            # Điều chỉnh kích thước ảnh tiền cảnh cho phù hợp với nền
            fgHeight, fgWidth = int(self.h / 4), int(self.w / 4)  # Đặt tiền cảnh nhỏ hơn nền
            self.foreground = cv.resize(fgImg, (fgWidth, fgHeight))
            self.center = self.currentCenter = (int(self.w / 2 - fgWidth / 2), int(self.h / 2 - fgHeight / 2))

            # Biên độ di chuyển tối đa của tiền cảnh
            self.xAmpl = self.w - (self.center[0] + fgWidth)
            self.yAmpl = self.h - (self.center[1] + fgHeight)
        else:
            self.foreground = None

        # Hình chữ nhật mặc định để mô phỏng khi không có tiền cảnh
        self.initialRect = np.array([
            (self.h // 2, self.w // 2),
            (self.h // 2, self.w // 2 + self.w // 10),
            (self.h // 2 + self.h // 10, self.w // 2 + self.w // 10),
            (self.h // 2 + self.h // 10, self.w // 2)
        ]).astype(int)
        self.currentRect = self.initialRect

    def getXOffset(self, time):
        return int(self.xAmpl * cos(time * self.speed))

    def getYOffset(self, time):
        return int(self.yAmpl * sin(time * self.speed))

    def getNextFrame(self):
        img = self.sceneBg.copy()

        if self.foreground is not None:
            # Tính toán vị trí di chuyển của tiền cảnh theo thời gian
            self.currentCenter = (
                self.center[0] + self.getXOffset(self.time),
                self.center[1] + self.getYOffset(self.time)
            )

            # Xác định tọa độ góc trên và dưới của vùng đặt tiền cảnh
            y0 = max(0, self.currentCenter[0])
            y1 = min(self.h, self.currentCenter[0] + self.foreground.shape[0])
            x0 = max(0, self.currentCenter[1])
            x1 = min(self.w, self.currentCenter[1] + self.foreground.shape[1])

            # Kiểm tra rằng vùng ghép có kích thước hợp lệ trước khi ghép
            if y1 > y0 and x1 > x0:
                # Xác định vùng tương ứng của foreground để ghép vào ảnh nền
                fg_y0 = 0 if y0 == self.currentCenter[0] else y0 - self.currentCenter[0]
                fg_x0 = 0 if x0 == self.currentCenter[1] else x0 - self.currentCenter[1]
                fg_y1 = fg_y0 + (y1 - y0)
                fg_x1 = fg_x0 + (x1 - x0)

                # Ghép ảnh tiền cảnh vào ảnh nền
                img[y0:y1, x0:x1] = self.foreground[fg_y0:fg_y1, fg_x0:fg_x1]
        else:
            # Di chuyển và biến dạng hình chữ nhật khi không có tiền cảnh
            self.currentRect = self.initialRect + np.int32(
                30 * cos(self.time * self.speed) + 50 * sin(self.time * self.speed)
            )
            if self.deformation:
                self.currentRect[1:3] += self.h // 20 * cos(self.time)
            cv.fillConvexPoly(img, self.currentRect, (0, 0, 255))

        self.time += self.timeStep
        return img

    def resetTime(self):
        self.time = 0.0

if __name__ == '__main__':
    # Đường dẫn đến ảnh nền và ảnh tiền cảnh
    backGr = cv.imread('DETECT_FACEMASK/data/graf1.png')
    fgr = cv.imread('DETECT_FACEMASK/data/box.png')

    # Kiểm tra xem ảnh có được tải thành công không
    if backGr is None or fgr is None:
        print("Lỗi: Không thể mở một hoặc cả hai tệp ảnh. Kiểm tra lại đường dẫn.")
    else:
        render = TestSceneRender(backGr, fgr)

        while True:
            img = render.getNextFrame()
            cv.imshow('img', img)

            ch = cv.waitKey(3)
            if ch == 27:  # Nhấn phím ESC để thoát
                break

        cv.destroyAllWindows()
