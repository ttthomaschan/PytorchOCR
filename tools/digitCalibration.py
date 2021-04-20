import re

string = "123,456.789.22"
digit = re.findall(r"\d",string)
# digit = re.findall(r"\d+\.?\d*",string)
splitdigit = string.split(',' and '.')
lastdigit = string.split('.')
exdigit = string.replace('.',',')
print(digit)
print(splitdigit)
print(lastdigit)
print(exdigit)