import cv2
import numpy as np

def max_filtering(N, I_temp):
    wall = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    wall[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)] = I_temp.copy()
    temp = np.full((I_temp.shape[0]+(N//2)*2, I_temp.shape[1]+(N//2)*2), -1)
    for y in range(0,wall.shape[0]):
        for x in range(0,wall.shape[1]):
            if wall[y,x]!=-1:
                window = wall[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num = np.max(window)
                temp[y,x] = num
    A = temp[(N//2):wall.shape[0]-(N//2), (N//2):wall.shape[1]-(N//2)].copy()
    return A

def min_filtering(N, A):
    wall_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    wall_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)] = A.copy()
    temp_min = np.full((A.shape[0]+(N//2)*2, A.shape[1]+(N//2)*2), 300)
    for y in range(0,wall_min.shape[0]):
        for x in range(0,wall_min.shape[1]):
            if wall_min[y,x]!=300:
                window_min = wall_min[y-(N//2):y+(N//2)+1,x-(N//2):x+(N//2)+1]
                num_min = np.min(window_min)
                temp_min[y,x] = num_min
    B = temp_min[(N//2):wall_min.shape[0]-(N//2), (N//2):wall_min.shape[1]-(N//2)].copy()
    return B

def background_subtraction(I, B):
    O = I - B
    norm_img = cv2.normalize(O, None, 0,255, norm_type=cv2.NORM_MINMAX)
    norm_img = 255 - norm_img
    return O,norm_img

def minimumBoxFilter(n, src):

    # Creates the shape of the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))

    # Applies the minimum filter with kernel NxN
    imgResult = cv2.erode(src, kernel)
    cv2.imwrite("/home/elimen/Data/Project/DocTab_Infer/test_results/min_26.jpg",imgResult)

    return imgResult


def maximumBoxFilter(n,src):
    
    # Creates the shape of the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))

    # Applies the maximum filter with kernel NxN
    imgResult = cv2.dilate(src, kernel)
    cv2.imwrite("/home/elimen/Data/Project/DocTab_Infer/test_results/max_26.jpg",imgResult)
    
    return imgResult


#############################################################################
if __name__ == "__main__":
    src_img = cv2.imread("/home/elimen/Data/Project/DocTab_Infer/test_images/test_29.jpg",0)
    N = 3
    max_img = maximumBoxFilter(N,src_img)
    min_img = minimumBoxFilter(N,src_img)
    _, dst = background_subtraction(src_img,min_img)
    cv2.imwrite("/home/elimen/Data/Project/DocTab_Infer/test_results/remove_26.jpg",dst)