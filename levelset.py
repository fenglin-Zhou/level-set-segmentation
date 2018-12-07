# -*- coding: UTF-8 -*-
from PIL import Image
import numpy as np
import cv2
import math
import ctypes
import matplotlib.pyplot as plt
#import levelset


showimg = None

class levelset_:
    CV_PI = 3.1415926535897932384626433832795
    # 基本参数
    m_iterNum = 50  # 迭代次数
    m_lambda1 = 1  # 全局项系数
    m_nu = 0.001 * 255 * 255  # 长度约束系数v
    m_mu = 1.0  # 惩罚项系数u
    m_timestep = 0.2  # 演化步长
    m_epsilon = 1.0  # 规则化参数
    # 过程参数
    m_mImage = None  # 源图像
    m_iCol = 0  # 图像宽度
    m_iRow = 0  # 图像高度
    m_depth = 0  # 水平集数据深度
    m_FGValue = 0.0  # 前景值
    m_BKValue = 0.0  # 背景值
    m_mPhi = None  # 水平集
    m_mDirac = None  # 狄拉克处理后的水平集
    m_mHeaviside = None  # 海氏函数处理后水平集：Н（φ）
    m_mCurv = None  # 水平集曲率κ=div(▽φ/|▽φ|)
    m_mK = None  # 惩罚项卷积核
    m_mPenalize = None  # 惩罚项中的▽<sup>2</sup>φ

    def __init__(self):
        self.m_BKValue = 0.0
#        print(self.m_iterNum)

    def initializePhi(self, img, rect):
        self.m_mImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("imgpy",self.m_mImage)
        #cv2.waitKey(0)
        size = img.shape
        self.m_iRow = size[0]
        self.m_iCol = size[1]
        self.m_depth = cv2.CV_32FC1
        #显式分配内存
        self.m_mPhi = np.zeros((self.m_iRow, self.m_iCol))
        self.m_mDirac = np.zeros((self.m_iRow, self.m_iCol))
        self.m_mHeaviside = np.zeros((self.m_iRow, self.m_iCol))
        # 初始化惩罚性卷积核
        self.m_mK = np.zeros((3, 3), dtype=float)
        self.m_mK[0][0] = 0.5
        self.m_mK[0][1] = 1
        self.m_mK[0][2] = 0.5
        self.m_mK[1][0] = 1
        self.m_mK[1][1] = -6
        self.m_mK[1][2] = 1
        self.m_mK[2][0] = 0.5
        self.m_mK[2][1] = 1
        self.m_mK[2][2] = 0.5

        c = 2
        print(rect.get_x(),rect.get_width())
        for i in range(self.m_iRow):
            for j in range(self.m_iCol):
                if (
                        i < rect.get_y() or i > rect.get_y() + rect.get_height() or j < rect.get_x() or j > rect.get_x() + rect.get_width()):
                    self.m_mPhi[i][j] = -c
                else:
                    self.m_mPhi[i][j] = c


    def Dirac(self):
        k1 = self.m_epsilon / self.CV_PI
        k2 = self.m_epsilon * self.m_epsilon
        for i in range(self.m_iRow):
            for j in range(self.m_iCol):
                self.m_mDirac[i][j] = k1 / (k2 + self.m_mPhi[i][j] * self.m_mPhi[i][j])
        #print(self.m_mDirac)
        #cv2.imshow("img", self.m_mDirac)
        #cv2.waitKey(0)

    def Heaviside(self):
        k3 = 2/self.CV_PI
        for i in range(self.m_iRow):
            for j in range(self.m_iCol):
                self.m_mHeaviside[i][j] = 0.5 * (1 + k3*math.atan(self.m_mPhi[i][j]/self.m_epsilon))
        #print(self.m_mHeaviside)
        #cv2.imshow("img", self.m_mHeaviside)
        #cv2.waitKey(0)

    def Curvature(self):
        dx = cv2.Sobel(self.m_mPhi, -1, 1, 0)
        dy = cv2.Sobel(self.m_mPhi, -1, 0, 1)
        for i in range(self.m_iRow):
            for j in range(self.m_iCol):
                val = math.sqrt(dx[i][j]*dx[i][j] + dy[i][j]*dy[i][j] + 1e-10)
                dx[i][j] = dx[i][j]/val
                dy[i][j] = dy[i][j]/val
        ddy = cv2.Sobel(dx, -1, 0, 1)
        ddx = cv2.Sobel(dy, -1, 1, 0)
        self.m_mCurv = ddx+ddy
        #print(self.m_mCurv)
        #cv2.imshow("img", self.m_mCurv)
        #cv2.waitKey(0)

    def BinaryFit(self):
        self.Heaviside()
        sumFG = 0.0
        sumBK = 0.0
        sumH = 0.0
        temp = self.m_mHeaviside
        temp2 = self.m_mImage
        fHeaviside = 0.0
        fFHeaviside = 0.0
        fImgValue = 0.0
        for i in range(self.m_iRow):
            for j in range(self.m_iCol):
                fImgValue = self.m_mImage[i][j]
                fHeaviside = float(self.m_mHeaviside[i][j])
                fFHeaviside = 1 - fHeaviside
                sumFG += fImgValue*fHeaviside
                sumBK += fImgValue*fFHeaviside
                sumH += fHeaviside
        self.m_FGValue = sumFG/(sumH + 1e-10)
        self.m_BKValue = sumBK/(self.m_iRow*self.m_iCol - sumH + 1e-10)
        #print(self.m_BKValue, self.m_FGValue)
        #cv2.waitKey(0)

    def EVolution(self):
        fCurv = 0.0
        fDirac = 0.0
        fPenalize = 0.0
        fImgValue = 0.0
        for i in range(self.m_iterNum):
            self.Dirac()
            self.Curvature()
            self.BinaryFit()
            self.m_mPenalize = cv2.filter2D(self.m_mPhi, -1, self.m_mK)
            #print(self.m_mPenalize)
            #cv2.imshow("m_mPenalize", self.m_mPenalize)
            #cv2.waitKey(0)
            for j in range(self.m_iRow):
                for k in range(self.m_iCol):
                    fCurv = self.m_mCurv[j][k]
                    fDirac = self.m_mDirac[j][k]
                    fPenalize = self.m_mPenalize[j][k]
                    fImgValue = ctypes.c_ubyte(self.m_mImage[j][k]).value
                    lengthTerm = self.m_nu*fDirac*fCurv
                    penalizeTerm = self.m_mu*(fPenalize - fCurv)
                    areaTerm = fDirac*self.m_lambda1*(-((fImgValue-self.m_FGValue)*(fImgValue-self.m_FGValue))+
                                                      ((fImgValue-self.m_BKValue)*(fImgValue-self.m_BKValue)))
                    self.m_mPhi[j][k] = self.m_mPhi[j][k]+self.m_timestep*(lengthTerm+penalizeTerm+areaTerm)
                    if j==100 and k==100:
                        print(areaTerm)
            global showimg
            #print(self.m_mPhi)
            #cv2.imshow("m_mphi", self.m_mPhi)
            #cv2.waitKey(0)
            showimg = cv2.cvtColor(self.m_mImage, cv2.COLOR_GRAY2BGR)
            mask = np.zeros((self.m_iRow, self.m_iCol))
            for j in range(self.m_iRow):
                for k in range(self.m_iCol):
                    if self.m_mPhi[j][k]>0:
                        mask[j][k] = 255
                    else:
                        mask[j][k] = 0
            #mask = self.m_mPhi
            mask = mask.astype(np.uint8)
            _, mask = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
            #print(mask)
            #cv2.imshow("mask", mask)
            #cv2.waitKey(0)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(mask, kernel)
            mask = cv2.erode(mask, kernel)
            __, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(showimg, contours, -1, (0, 0, 255), 1)
            cv2.imshow("img", showimg)
            cv2.waitKey(1)


if __name__ == "__main__":
    img = cv2.imread('./5.png')
    size = img.shape
    rect = plt.Rectangle((0, 0), size[1], size[0])
    #print(rect)
    #initializePhi(img, rect)
    #Dirac()
    level = levelset_()
    level.initializePhi(img, rect)
    level.EVolution()
    cv2.imshow("img",showimg)
    cv2.waitKey(0)
