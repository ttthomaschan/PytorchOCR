import cv2
import os

g = os.walk("/home/elimen/Data/Project/test_image_pdf/angle_rotate/")
res_path = "/home/elimen/Data/Project/test_image_pdf/interpol/"
for path,dir_list,file_list in g:  
    for file_name in file_list:  
        imgfile = os.path.join(path, file_name)
        srcImg = cv2.imread(imgfile)
        h, w, _= srcImg.shape
        interpol_img = cv2.resize(srcImg, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(res_path + file_name.split('.')[0] + "_Interpol.jpg", interpol_img)
