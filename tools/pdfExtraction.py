import os
import fitz

def pdf2img(pdfPath,page,zoom_x=10,zoom_y=10,logMode=False):

    pdf = fitz.open(pdfPath)
    rotation_angle = 0
    pages = pdf.pageCount

    if page in range(1,pages+1):
        pg = pdf[page - 1]
        trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation_angle)
        pm = pg.getPixmap(matrix=trans, alpha=False)
        if logMode:
            pm.writePNG(pdfPath.split('.')[0]+str(page)+".png")
        getpngdata = pm.getImageData(output="png")
        ## decode to np.uint8
        img_array = np.frombuffer(getpngdata, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR)
    else:
        '''返回错误码'''

    pdf.close()
    return img_cv


if "__main__" == "__name__"
    pdfPath = '/home/elimen/Data/dbnet_pytorch/test_pdf/zd.pdf'
    img = pdf2img(pdfPath,1）