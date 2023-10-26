import cv2
import cv2 as cv
# import numpy as np
# import os
# import time

names = ['My','pujing','pujing','pujing','pujing','pujing','pujing',]

# 打开摄像头
def video_demo():
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()

    while (True):
        Read_ref, Read_frame = capture.read()
        Read_Image_Data(Read_frame)
        cv.imshow("image", Read_frame)
        Key_Data = cv.waitKey(1) & 0xff
        if Key_Data == 27:
            capture.release()
            break


def Read_Image_Data(image_Data):
    # 读取识别的数据
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./Data/trainer/trainer.yml')
    # 读取图像
    # image_Data = cv.imread("image.jpg")
    # 把图像转换为灰度图
    gray_Data = cv.cvtColor(image_Data, cv2.COLOR_BGR2GRAY)
    # 加载人脸识别器
    face_cascade = cv.CascadeClassifier(r"./Data/haarcascade_frontalface_alt2.xml")
    # 检测灰度图中的所有面孔
    Face_Out_Data = face_cascade.detectMultiScale(gray_Data,1.01,5,0,(100,100),(300,300))
    # 对识别出来的图像进行画框
    for x, y, width, height in Face_Out_Data:
        # 人脸识别
        id, confidence = recognizer.predict(gray_Data[y:y + height, x:x + width])
        # 这里的color是 蓝 黄 红，与rgb相反，thickness设置宽度
        cv.rectangle(image_Data, (x, y), (x + width, y + height), color=(255,0,255), thickness=3)
    # 保存输出图像
    # cv.imwrite("beauty_detected.jpg", image_Data)
        name = names[id]
        cv.putText(image_Data,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

if __name__ == '__main__':
    video_demo()
    # Read_Image_Data()
    cv.destroyAllWindows()
