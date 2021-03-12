import pdfplumber
import pandas as pd

pdf = pdfplumber.open('/home/elimen/Data/dbnet_pytorch/test_pdf/yy.pdf')

first_page = pdf.pages[0]
sec_page = pdf.pages[1]

text = sec_page.extract_text()

table = first_page.extract_table()
table_df = pd.DataFrame(table)
table_df.to_excel('/home/elimen/Data/dbnet_pytorch/test_pdf/pdf2excel.xlsx')

