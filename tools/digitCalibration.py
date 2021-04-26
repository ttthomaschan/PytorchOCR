import re

def digitCalibration(string,decimals):
    #1. 判断是否数字串
    is_numSeries = False
    calibratedRes=""
    if re.match('[0-9]', string):
        is_numSeries = True
    else:
        is_numSeries = False
        return string
    
    #2. 判断是否有符号
    is_containSbl = False
    calibratedRes=""
    if re.search('[\,\.\-]', string):
        is_containSbl = True
    else:
        is_containSbl = False
        return string

    #3. 判断最后一个标点分隔的数字位数
    if is_numSeries and is_containSbl and len(re.split(r'[,|.]',string)[-1]) == decimals:
        calibratedRes = string.replace('.',',')
        calibratedRes = calibratedRes[:-3] + "." + calibratedRes[-2:]
        return calibratedRes
    else:
        return string
    

if "__main__" == __name__:
    string = "123,456.789.22"  #123,456.789.22  #网址www.163.com
    res = digitCalibration(string,2)
    print(res)