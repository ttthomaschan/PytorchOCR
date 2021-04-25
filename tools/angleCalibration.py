import cv2
import numpy as np

'''1. 检测倾斜角度，旋转矫正图片'''
class calibrateRotatedImage():
    # 输入检查 --> 自适应二值化 --> 形态学过滤横线
    def __init__(self,srcImg):
        self.src = srcImg
        self.grey = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        self.binary = cv2.adaptiveThreshold(self.grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
        ## 识别横线:
        # 自适应获取核值
        rows, cols = self.binary.shape
        scale = 20
        scale2 = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale2, 1))
        eroded_col = cv2.erode(self.binary, kernel, iterations=1)
        self.dilated_col = cv2.dilate(eroded_col, kernel2, iterations=1)

    # 弧度 --> 角度
    def DegreeTrans(self,theta):
        degree = theta / np.pi * 180
        return degree

    def CalcDegree(self):
        cannyImage = cv2.Canny(self.dilated_col, 50, 200, 3)
        self.lineimage = self.src.copy()
    
        # 通过霍夫变换检测直线
        # 第4个参数就是阈值，阈值越大，检测精度越高
        lines = cv2.HoughLines(cannyImage, 1, np.pi/180, 200)
        # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
        sum = 0
        # 依次画出每条线段
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                # print("theta:", theta, " rho:", rho)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(round(x0 + 1000 * (-b)))
                y1 = int(round(y0 + 1000 * a))
                x2 = int(round(x0 - 1000 * (-b)))
                y2 = int(round(y0 - 1000 * a))
                # 只选角度最小的作为旋转角度
                sum += theta
                cv2.line(self.lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
        
        # 对所有角度求平均，这样做旋转效果会更好
        average = sum / len(lines)
        self.angle = self.DegreeTrans(average) - 90
        # print("angle:{}".format(angle))
        return self.angle

    def rotateImage(self):
        # 旋转中心为图像中心 ==> 可以改进为 以图形重心为旋转中心
        h, w = self.src.shape[:2]
        # 计算二维旋转的仿射变换矩阵
        RotateMatrix = cv2.getRotationMatrix2D((w/2.0, h/2.0), self.angle, 1)
        # 仿射变换，背景色填充为白色
        rotatedimg = cv2.warpAffine(self.src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
        return rotatedimg
    
    # 计算倾斜角度 --> 计算仿射变换矩阵 --> 返回旋转矫正后的图片
    def __call__(self):
        rotateAngle = self.CalcDegree() 
        rotateImage = self.rotateImage()
        return rotateAngle, rotateImage

        
    # 检查横线是否正确，用于调试
    def check_horizontal_lines(self,logPath):
        cv2.imwrite(logPath+"/mophologyLines.jpg", self.dilated_col)
        cv2.imwrite(logPath+"/cannyLines.jpg", self.lineimage)
        return True

'''2. 计算图形几何中心（重心）[优化方向]'''
class CalculateBarycenter():
    ## 参数格式： pts = [tl,tr,br,bl]
    def __init__(self, pts):
        self.top_middle_pt = 0
        self.right_middle_pt = 0
        self.bottom_middle_pt = 0
        self.left_middle_pt = 0
        self.vertical_line_length = 0
        self.horizontal_line_length = 0

    def __call__(self):
        return True


#########################################################################3333

if __name__ == "__main__":
    imgPath = "/home/elimen/Data/dbnet_pytorch/test_images/rotated02.jpg"
    logPath = "/home/elimen/Data/dbnet_pytorch/test_results/angleCalibration"
    srcImg = cv2.imread(imgPath,1)

    calRotImg = calibrateRotatedImage(srcImg)
    angle, rotated_img = calRotImg()
    calRotImg.check_horizontal_lines(logPath)
    print(angle)
    cv2.imwrite(logPath+"/calibratedImg.jpg", rotated_img)



