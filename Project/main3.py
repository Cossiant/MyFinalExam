import cv2
import os
import sys
from PIL import Image
import numpy as np
def getImageAndLabels(path):
    facesSamples=[]
    ids=[]
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagePaths)
    # 检测人脸
    face_detector = cv2.CascadeClassifier('./Data/haarcascade_frontalface_default.xml')

    # 遍历列表中的图片
    for imagePath in imagePaths:
        # 打开图片
        PIL_img = Image.open(imagePath).convert('L')
        # 将图像转换为数组
        img_numpy = np.array(PIL_img,'uint8')
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id
        print(os.path.split(imagePath))
        id = int(os.path.split(imagePath)[1].split('.')[0])
        for x,y,w,h in faces:
            facesSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return facesSamples,ids


if __name__== '__main__':
    #图片路径
    path = './data/MyPath/'
    #获取图像 数组和id标签
    faces,ids=getImageAndLabels(path)
    #获取循环对象
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
    #保存文件
    recognizer.write('./Data/trainer/trainer.yml')