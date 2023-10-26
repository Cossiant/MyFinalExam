import cv2 as cv
import numpy as np
from PIL import Image
import os

def getImageAndLabels(Path):
    # 人脸数据
    facesSamples = []
    # 存储姓名数据
    ids = []
    # 存储图片信息
    imagePaths = [os.path.join(Path,f)for f in os.listdir(Path)]
    # 加载分类器
    face_detector = cv.CascadeClassifier('./Data/haarcascade_frontalface_default.xml')
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')

        img_numpy = np.array(PIL_img,'uint8')

        faces = face_detector.detectMultiScale(img_numpy)

        id = int(os.path.split(imagePath)[1].split('.')[0])

        for x,y,w,h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h,x:x+w])
    print('id:',id)
    print('fs:',facesSamples)
    return facesSamples,ids


if __name__ == '__main__':
    Path = './Data/MyPath/'
    faces,ids = getImageAndLabels(Path)
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
    recognizer.write('./Data/trainer/trainer.yml')