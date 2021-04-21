import cv2

'''1. 检测表格外角点'''
class ExtractAngularPoint():
    # 自适应二值化 --> 边缘检测 --> 边界筛选
    def __init__(self,srcImg,logPath):
        self.src = srcImg
        self.thres_img = cv2.adaptiveThreshold(self.src, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        cv2.imwrite(logPath+"/thresImg.jpg", thres_img)
        self.canny_img = cv2.Canny(thres_img,100,200)
        v2.imwrite(logPath+"/cannyImg.jpg", canny_img)
        contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.contours = contours

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

    

if __name__ == "__main__":
    imgPath = "/home/elimen/Data/dbnet_pytorch/test_images/Red_thres.jpg"
    logPath = "/home/elimen/Data/dbnet_pytorch/test_results"
    srcImg = cv2.imread(imgPath,CV_8UC1)
    extract_angular = ExtractAngularPoint(srcImg,logPath)
