import cv2
import numpy as np

'''1. 检测表格外角点'''
class ExtractAngularPoint():
    # 自适应二值化 --> 边缘检测 --> 边界筛选
    def __init__(self,srcImg,logPath):
        self.src = srcImg
        self.grey = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        
        self.thres_img = cv2.adaptiveThreshold(self.grey, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        cv2.imwrite(logPath+"/thresImg.jpg", self.thres_img)
        
        self.canny_img = cv2.Canny(self.thres_img,30,90)
        cv2.imwrite(logPath+"/cannyImg.jpg", self.canny_img)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_img = cv2.dilate(self.canny_img, kernel, iterations=1)
        eroded_img = cv2.erode(dilated_img, kernel, iterations=1)
        cv2.imwrite(logPath+"/erodeImg.jpg", eroded_img)
        
        _, contours, hierarchy = cv2.findContours(self.canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.contours = contours
        
        area=[]
        for k in range(len(contours)):
	        area.append(cv2.contourArea(contours[k]))
        max_idx = np.argmax(np.array(area))
        cv2.drawContours(self.src,contours,max_idx,(0,255,0),5) 
        cv2.imwrite(logPath+"/contours.jpg", self.src)

    # 计算外角点,返回4个角点坐标
    def __call__(self):

        return True
    
    # 检查边框是否正确，用于调试
    def check_max_contour(self,path):
        return True

'''2. 计算图形几何中心（重心）'''
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

'''3. 透视变换矫正图形'''
# class AngleCalibration():
#     def __init__(self,src,dst):


#     def __call__():



def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res

def CalcDegree(srcImage):
    print()
    if len(srcImage.shape) == 3:
        midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    elif len(srcImage.shape) == 2:
        midImage = srcImage
    dstImage = cv2.Canny(midImage, 50, 200, 3)
    lineimage = srcImage.copy()
 
    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    lines = cv2.HoughLines(dstImage, 1, np.pi/180, 200)
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
            cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
    
    logPath = "/home/elimen/Data/dbnet_pytorch/test_results/angleCalibration"
    cv2.imwrite(logPath+"/lines.jpg", lineimage)
    # 对所有角度求平均，这样做旋转效果会更好
    average = sum / len(lines)
    angle = DegreeTrans(average) - 90
    print("angle:{}".format(angle))
    return angle

def rotateImage(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w/2.0, h/2.0), degree, 1)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate


if __name__ == "__main__":
    imgPath = "/home/elimen/Data/dbnet_pytorch/test_images/rotated.jpg"
    logPath = "/home/elimen/Data/dbnet_pytorch/test_results/angleCalibration"
    #extract_angular = ExtractAngularPoint(dilated_col,logPath)

    srcImg = cv2.imread(imgPath)
    gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)

    rows, cols = binary.shape
    scale2 = 15
    scale = 20
    # 自适应获取核值
    # 识别横线:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale2, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated_col = cv2.dilate(eroded, kernel1, iterations=1)
    cv2.imwrite(logPath+"/1_横向形态学.jpg", dilated_col)
    
    angle1 = CalcDegree(dilated_col)
    print(angle1)
    rotated_img = rotateImage(dilated_col,angle1)
    angle2 = CalcDegree(rotated_img)
    print(angle2)


    _ = rotateImage(srcImg,angle1)
    rotated_img = rotateImage(srcImg,angle1)
    cv2.imwrite(logPath+"/rotated02.jpg", rotated_img)



