import re

string = "123.456,789.22"  #123,456.789.22 #网址www.163.com
#1. 判断是否数字串
is_numSeries = False
res=""
if re.match('[0-9\,\.\-]', string):
    is_numSeries = True
else:
    is_numSeries = False

#2. 判断最后一个标点分隔的数字位数
if len(re.split(r'[,|.]',string)[-1]) == 2:
    res = string.replace('.',',')
    res = res[:-3] + "." + res[-2:]

print(res)
