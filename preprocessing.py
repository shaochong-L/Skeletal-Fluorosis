import cv2 as cv
import numpy as np
import math
import cv2
import imutils
from PIL import Image
import os
from matplotlib import pyplot as plt

img_path = "D:/19S004085/project1/bone/preprocessing/"
trans_path = "D:/19S004085/project1/bone/preprocessing/"
result_path = "D:/19S004085/project1/bone/preprocessing/"

# def pre_processing(name):
#     # 直方均衡
#     img = cv2.imread(img_path + name, 0)
#     equ = cv2.equalizeHist(img)
#     cv2.imwrite(trans_path + "1zhifang.jpg", equ)
#
#     # 光滑
#     # 用 np.hstack，将多个 均值平滑处理后的图像水平合并起来
#     blured = cv2.blur(equ, (20, 20))
#     cv2.imwrite(trans_path + "2guanghua.jpg", blured)
#
#     # 二值
#     ret, binary = cv.threshold(blured, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     cv.imwrite(trans_path + "3erzhi.jpg", binary)
#
#     # 腐蚀
#     kernel1 = np.ones((20, 20), np.uint8)
#     erosion1 = cv2.erode(binary, kernel1, iterations=1)
#     cv.imwrite(trans_path + "4fushi1.jpg", erosion1)
#
#     #膨胀
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
#     dilation = cv2.dilate(erosion1, kernel2, iterations=20)
#     cv.imwrite(trans_path + "5pengzhang.jpg", dilation)
#
#     #腐蚀
#     kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
#     erosion = cv2.erode(dilation, kernel1, iterations=10)
#     cv.imwrite(trans_path + "6fushi.jpg", erosion)
#
#
#
#     # 计算角度
#     image = cv.imread(trans_path + "pengzhang.jpg")
#     edges = cv.Canny(dilation, 50, 310)  # apertureSize参数默认其实就是3  # 50 310
#     # cv.imshow("edges", edges)
#     # edge = Image.fromarray(edges)
#     # edge.save("edge.jpeg")
#     lines = cv.HoughLines(edges, 1, np.pi / 180, 68)  # 68
#     # l1 = lines[:, 0, :]
#     # print(l1)
#     mink = float('inf')
#     maxk = -float('inf')
#     for line in lines:
#         rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
#         a = np.cos(theta)  # theta是弧度
#         b = np.sin(theta)
#         x0 = a * rho  # 代表x = r * cos（theta）
#         y0 = b * rho  # 代表y = r * sin（theta）
#         x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
#         y1 = int(y0 + 1000 * a)  # 计算起始起点纵坐标
#         x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
#         y2 = int(y0 - 1000 * a)  # 计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
#         dx = x2 -x1
#         if abs (dx)  <= 1e-2:
#             dx = 0.00001
#         k = (y2 - y1) / dx
#         if k > maxk:
#             maxk = k
#             xmax1 = x1
#             ymax1 = y1
#             xmax2 = x2
#             ymax2 = y2
#             lineMax = line
#             thetamax = theta
#         if k < mink:
#             mink = k
#             xmin1 = x1
#             ymin1 = y1
#             xmin2 = x2
#             ymin2 = y2
#             lineMin = line
#             thetamin = theta
#     cv.line(image, (xmax1, ymax1), (xmax2, ymax2), (255, 0, 0), 2)  # 点的坐标必须是元组，不能是列表。
#     cv.line(image, (xmin1, ymin1), (xmin2, ymin2), (255, 0, 0), 2)  # 点的坐标必须是元组，不能是列表。
#     theta = 90 - (thetamax + thetamin) / 2 / math.pi * 180
#     print(name,theta)
#
#     # 旋转
#     image = cv2.imread(img_path + name)
#     rotated = imutils.rotate(image, -theta)
#     cv2.imwrite(trans_path + "rotated.bmp", rotated)
#
#     #裁剪
#     im = Image.open(trans_path + "rotated.bmp")
#     width = im.width
#     height = im.height
#     middle_width = int(width/2)
#     middle_height = int(height/2)
#
#     '''
#     裁剪：传入一个元组作为参数
#     元组里的元素分别是：（距离图片左边界距离x， 距离图片上边界距离y，距离图片左边界距离+裁剪框宽度x+w，距离图片上边界距离+裁剪框高度y+h）
#     '''
#
#     x = middle_width - 750
#     y = middle_height - 250
#     w = 1500
#     h = 500
#     region = im.crop((x, y, x + w, y + h))
#     region.save(result_path + name)

def pre_processing2(name):
    a = cv.imread(img_path + name, 0)
    a = cv.GaussianBlur(a, (5, 5), 0)

    # hist = cv.calcHist([a], [0], None, [256], [0, 256])
    # plt.subplot(132)
    # plt.plot(hist)
    ret, dst = cv.threshold(a, 100, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
    print(dst)
    cv.imwrite(img_path+"1_erzhi.jpg",dst)

    dst = cv.medianBlur(dst, 75)
    s = cv.getStructuringElement(cv.MORPH_CROSS, (10, 10))
    dst = cv.erode(dst, s, borderType=cv.BORDER_CONSTANT, borderValue=0)
    cv.imwrite(img_path + "2_lvbo.jpg",dst)

    rows, cols = dst.shape[:2]
    i, c, h = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    dsts = [dst, dst, dst]
    dst1 = cv.merge(dsts)

    ch = c[-1]
    [vx, vy, x, y] = cv.fitLine(ch, cv.DIST_L2, 0, 1, 0.01)

    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    theta = math.atan(vy / vx) / math.pi * 180
    print(name,theta)

    #旋转
    image = cv2.imread(img_path + name)
    rotated = imutils.rotate(image, theta)
    cv2.imwrite(trans_path + "4_rotated.jpg", rotated)

    #裁剪
    im = Image.open(trans_path + "4_rotated.jpg")
    width = im.width
    height = im.height
    middle_width = int(width/2)
    middle_height = int(height/2)

    '''
    裁剪：传入一个元组作为参数
    元组里的元素分别是：（距离图片左边界距离x， 距离图片上边界距离y，距离图片左边界距离+裁剪框宽度x+w，距离图片上边界距离+裁剪框高度y+h）
    '''

    x = middle_width - 375
    y = middle_height - 250
    w = 750
    h = 500
    region = im.crop((x, y, x + w, y + h))
    region.save(result_path + "5_final.jpg")

# for filename in os.listdir(img_path):

pre_processing2("11.jpg")