import cv2
import numpy as np

def colorSeparation(srcImg):
    # check image
    if len(srcImg.shape) != 3:
        return -1
    else:
        cols, rows, _ = srcImg.shape
        img = srcImg
    
    B,G,R = cv2.split(img)
    # Red channel thresholding
    _, img_RedThresh = cv2.threshold(R,200,255,cv2.THRESH_BINARY)
    
    # Erode
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)) 
    img_Erode = cv2.erode(img_RedThresh, element)
    img_Dilate = cv2.dilate(img_Erode, element)

    # Merge
    # zeros = np.zeros(srcImg.shape[:2], dtype = "uint8")          
    img_Merged = cv2.cvtColor(img_RedThresh, cv2.COLOR_GRAY2BGR)   
    
    return img_RedThresh, img_Merged

###############################################################################
if __name__ == "__main__":
    
    import os
    g = os.walk("/home/elimen/Data/Project/test_image_pdf/test_image/")
    res_path = "/home/elimen/Data/Project/test_image_pdf/color_split/"
    for path,dir_list,file_list in g:  
        for file_name in file_list:  
            imgfile = os.path.join(path, file_name)
            srcImg = cv2.imread(imgfile)
            img_R, img_Merged = colorSeparation(srcImg)
            
            # cv2.imwrite(res_path + file_name + "_R.jpg", img_R)
            cv2.imwrite(res_path + file_name.split('.')[0] + "_Merged.jpg", img_Merged)

