import cv2
import cv2 as cv
# import numpy as np
# import os
# import time


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
    # 读取图像
    # image_Data = cv.imread("image.jpg")
    # 把图像转换为灰度图
    gray_Data = cv.cvtColor(image_Data, cv2.COLOR_BGR2GRAY)
    # 加载人脸识别器
    face_cascade = cv.CascadeClassifier(r"./Data/haarcascade_frontalface_default.xml")
    # 检测灰度图中的所有面孔
    Face_Out_Data = face_cascade.detectMultiScale(gray_Data)
    # 对识别出来的图像进行画框
    for x, y, width, height in Face_Out_Data:

        # 这里的color是 蓝 黄 红，与rgb相反，thickness设置宽度
        cv.rectangle(image_Data, (x, y), (x + width, y + height), color=(255,255, 0), thickness=5)
    # 保存输出图像
    # cv.imwrite("beauty_detected.jpg", image_Data)


if __name__ == '__main__':
    video_demo()
    # Read_Image_Data()
    cv.destroyAllWindows()
