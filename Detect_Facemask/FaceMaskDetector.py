from email.mime import image
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv
import cv2
import video
import datetime, time
from PIL import Image, ImageOps
import random
import string
import pygame  # Thêm pygame để phát âm thanh cảnh báo

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURRENT_DIR)

# Đường dẫn tới file âm thanh cảnh báo
WARNING_SOUND = os.path.join(CURRENT_DIR, "canhbao.mp3")

# Khởi tạo pygame mixer
pygame.mixer.init()

class FaceMask:

    LABELS = []
    cascade = None
    model = None 
    size = (224, 224)

    
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(os.path.join(CURRENT_DIR, "cascade.xml"))
        if(self.cascade.empty()):
            print("cascade empty")

        self.getLabels()

        modelFile = os.path.join(CURRENT_DIR, "keras_model.h5")
        if(os.path.exists(modelFile)):
            self.model = tf.keras.models.load_model(modelFile)
        
        #data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        np.set_printoptions(suppress=True)

    ####################################################################################################
    
    def getLabels(self):
        with open(os.path.join(CURRENT_DIR, "labels_facemask.txt"), 'r') as file:
            for x in file:
                self.LABELS.append(str(x).replace("\n", ""))
        print(self.LABELS)

    ####################################################################################################

    def play_warning(self):
        """Phát âm thanh cảnh báo."""
        pygame.mixer.music.load(WARNING_SOUND)
        pygame.mixer.music.play()

    ####################################################################################################

    def TFpredictImgPath(self, imgePath):
        pilImg = image.load_img(imgePath)    

        return self.TFpredictPilImg(pilImg)

    ####################################################################################################

    def TFpredictPilImg(self, pilImg):
        if(self.model == None):
            print("model is null")
            return None
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        image = ImageOps.fit(pilImg, self.size, Image.LANCZOS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        predictions = self.model.predict(data)
        result = np.argmax(predictions)
        return result

    ####################################################################################################

    def PredictMat(self, mat):
        img = cv.cvtColor(mat, cv.COLOR_BGR2RGB)    
        img = cv.resize(img, self.size)
        img_pil = Image.fromarray(img)

        result = self.TFpredictPilImg(img_pil)
        return result

    ####################################################################################################

    def DetectFaceInFrame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)

        newWidth = int(gray.shape[1] /2)
        newHeight = int(gray.shape[0] /2)

        gray = cv2.resize(gray, (newWidth, newHeight))
        rects = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(20,20))

        if(len(rects) > 0):
            rects[:,2:] += rects[:,:2]

        for r in rects:
            r[0] *= 2
            r[1] *= 2
            r[2] *= 2
            r[3] *= 2
            

        return rects

    ####################################################################################################

    def CropMat(self, frame, rect):
        x = rect[0]
        y = rect[1]
        w = rect[2] - x
        h = rect[3] - y

        frame = frame[y:h, x:w]
        return frame

    ####################################################################################################

    def GenerateRandomString(self):
        return ''.join(random.choices(string.ascii_lowercase + "_" + string.ascii_uppercase +  string.digits, k=10))

    ####################################################################################################

    def DetectMask(self, frame):
        startTime = time.time()
        saveImage = False #to debug

        rects = self.DetectFaceInFrame(frame)
        arr = []
        if(len(rects) > 0):
            rects[:,2:] += rects[:,:2]
            for rect in rects:
                #print(rect)
                matFace = self.CropMat(frame, rect)                
                predicted = self.PredictMat(matFace)
                result = self.LABELS[predicted]

                color = (0, 0, 255)
                
                
                x1 = rect[0]
                y1 = rect[1]
                x2 = rect[2] - x1
                y2 = rect[3] - y1

                elapsed = time.time() - startTime

                if(saveImage):
                    cv2.imwrite(result +"\\" + datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S") + "_" + GenerateRandomString() + ".jpg", matFace) # type: ignore

                arr.append(result)

                if(result == "Mask"):
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    
                elif(result == "Hand" or result == "No mask" or result == "Wrong"):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    self.play_warning()  # Gọi âm thanh cảnh báo

                if(result != "Nothing"):
                    cv.putText(frame, str(result), (10,60), cv.FONT_HERSHEY_PLAIN, 3, color, thickness = 2)
                    cv.putText(frame, "{:10.2f} s".format(elapsed), (300,60), cv.FONT_HERSHEY_PLAIN, 3, color, thickness = 2)
        else:
            if(saveImage):
                cv2.imwrite("noface\\" + datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S") + "_" + GenerateRandomString() + ".jpg", frame) # type: ignore

        return frame, arr

####################################################################################################


faceMask = FaceMask()

if __name__ == '__main__':
    cap = video.create_capture(0)
    while True:
        
        _ret, frame = cap.read()
        frame, arr = faceMask.DetectMask(frame)

        cv.imshow('frame', frame)
        ch = cv.waitKey(20)
        if ch == 27:
            break