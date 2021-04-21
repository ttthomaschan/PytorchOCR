import pdfplumber
import pandas as pd

#pdf = pdfplumber.open('/home/elimen/Data/dbnet_pytorch/test_pdf/mt52.pdf')

# first_page = pdf.pages[0]
# #sec_page = pdf.pages[1]

# #text = sec_page.extract_text()

# table = first_page.extract_table()
# table_df = pd.DataFrame(table)
# table_df.to_excel('/home/elimen/Data/dbnet_pytorch/test_pdf/pdf2excel.xlsx')

import os
import fitz

pdfPath = '/home/elimen/Data/dbnet_pytorch/test_pdf/zd.pdf'
pdf = fitz.open(pdfPath)
pages = pdf.pageCount

zoom_x = 10
zoom_y = 10
rotation_angle = 0
for pg in range(0, pdf.pageCount):
    page = pdf[pg]
    trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation_angle)
    pm = page.getPixmap(matrix=trans, alpha=False)
    #pm.writePNG(pdfPath.split('.')[0]+str(pg)+".png")
    getpngdata = pm.getImageData(output="png")
    # decode to np.uint8
    image_array = np.frombuffer(getpngdata, dtype=np.uint8)
    img_cv = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)

pdf.close()
