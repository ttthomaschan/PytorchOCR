import cv2
import numpy as np
import math
import xlsxwriter

class TabRecognition:
	def __init__(self,image,logPath):
		'''
		image: original image
		target: the combine of box and content
		'''
		self.image = image
		self.logPath = logPath

	'''
	函数 islianjie（） 用于判断两点之间是否有连接。原理：两点之间 取多条 垂直的 横截线段，如果有像素值为0的，可以判断这两点是断开的。
	'''
	def islianjie(self,p1,p2,img): # 坐标p的格式是先y轴后x轴
		if p1[0]==p2[0]:   # y坐标相同，在同一横线
			for i in range(min(p1[1],p2[1]),max(p1[1],p2[1])+1):
				if sum( [ img[j,i] for j in range( max(p1[0]-5, 0), min(p1[0]+5, img.shape[0]) ) ] )==0: # img mask 格式也是先y后x
					return False
			return True

		elif p1[1]==p2[1]:  # x坐标相同，在同一竖线
			tmpsum = 0
			for i in range(min(p1[0],p2[0]), max(p1[0],p2[0])+1):   # y轴变化范围 
				if sum( [img[i,j] for j in range(max(p1[1]-5,0), min(p1[1]+5,img.shape[1])) ] ) == 0:   # x轴变化范围
					return False
			return True

		else:
			return False

	def crossingpointDetection(self):
		raw = self.image
		gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)    #  转换为灰度图片
		binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
		
		# 自适应获取核值
		rows, cols = binary.shape
		scale = 20
		scale2 = 15
		# 形态学处理，识别横线:
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
		kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale2, 1))
		eroded = cv2.erode(binary, kernel, iterations=1)
		dilated_col = cv2.dilate(eroded, kernel1, iterations=1)
		cv2.imwrite(self.logPath+"1_横向形态学.jpg", dilated_col)

		# 形态学处理，识别竖线：
		# scale = 40#scale越大，越能检测出不存在的线
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // 35))  # scale
		kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // 20))  # scale2
		eroded = cv2.erode(binary, kernel, iterations=1)
		dilated_row = cv2.dilate(eroded, kernel2, iterations=1)
		cv2.imwrite(self.logPath+"2_竖向形态学.jpg", dilated_row)

		# 将识别出来的横竖线合起来
		bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)  # 对二值图进行 与操作，即可求得交点
		cv2.imwrite(self.logPath+"3_横向竖向交点.jpg", bitwise_and)

		# 标识表格轮廓
		merge = cv2.add(dilated_col, dilated_row)
		ret,binary = cv2.threshold(merge, 127, 255, cv2.THRESH_BINARY)
		self.merge = merge.copy()
		cv2.imwrite(self.logPath+"4_横竖交点阈值化.jpg", binary)

		ys, xs = np.where(bitwise_and > 0)

		'''
		关键点： 利用相邻位置信息，过滤重复直线。输出为： 交点的横纵坐标数组y_point_arr， x_point_arr
		'''
		# 交点的横纵坐标数组
		y_point_arr = []
		x_point_arr = []
		# 通过排序，排除掉相近的像素点，只取相近值的中位数点
		# 这个3就是两个像素点的距离，不是固定的，根据不同的图片会有调整，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
		i = 0
		sort_x_point = np.sort(xs)
		tmpIndex = 0
		for i in range(len(sort_x_point) - 1):
			if sort_x_point[i + 1] - sort_x_point[i] > 3:
				midIndex = (tmpIndex + i) // 2
				x_point_arr.append(sort_x_point[midIndex])
				tmpIndex = i + 1
			i = i + 1
		x_point_arr.append(sort_x_point[i])  # 要将最后一个点加入

		i = 0
		sort_y_point = np.sort(ys)
		tmpIndex = 0
		for i in range(len(sort_y_point) - 1):
			if sort_y_point[i + 1] - sort_y_point[i] > 3:
				midIndex = (tmpIndex + i) // 2
				y_point_arr.append(sort_y_point[midIndex])
				tmpIndex = i + 1
			i = i + 1
		y_point_arr.append(sort_y_point[i])

		self.y_crossingpoint_arr = y_point_arr
		self.x_crossingpoint_arr = x_point_arr
		return y_point_arr, x_point_arr

	
	def cellRecognition(self):
		y_point_arr = self.y_crossingpoint_arr
		x_point_arr = self.x_crossingpoint_arr 
		
		# 计算所有可能的长和宽
		h_list = [y_point_arr[i+1]-y_point_arr[i] for i in range(len(y_point_arr)-1)]
		w_list = [x_point_arr[i+1]-x_point_arr[i] for i in range(len(x_point_arr)-1)]

		lt_list_x = x_point_arr[:-1]  # 取前面的n-1个值，最后一个不取
		lt_list_y = y_point_arr[:-1]
		rd_list_x = x_point_arr[1:]   # 从第2个值开始，第一个值不取，共n-1个值
		rd_list_y = y_point_arr[1:]

		d={}
		for i in range(len(lt_list_x)):
			for j in range(len(lt_list_y)):
				d['cell_{}_{}'.format(i,j)] = cell( [lt_list_x[i],lt_list_y[j]], [rd_list_x[i],rd_list_y[j]], [lt_list_x[i],lt_list_y[j]])

		for i in range(len(lt_list_x)):
			for j in range(len(lt_list_y)):
				## p点格式为(y,x)。假设 左上角 lt(y1,x1), 右下角 rd(y2,x2) ==> 左下角 p1(y2,x1), 右上角 p3(y1,x2)
				p1 = [d['cell_{}_{}'.format(i,j)].rd[1], d['cell_{}_{}'.format(i,j)].lt[0]]  #左下点 
				p2 = [d['cell_{}_{}'.format(i,j)].rd[1], d['cell_{}_{}'.format(i,j)].rd[0]]  #右下点 
				p3 = [d['cell_{}_{}'.format(i,j)].lt[1], d['cell_{}_{}'.format(i,j)].rd[0]]  #右上点
				## 查看两点之间是否连接，确定单元格归属
				if not self.islianjie(p1,p2,self.merge):
					d['cell_{}_{}'.format(i,j+1)].belong = d['cell_{}_{}'.format(i,j)].belong
				if not self.islianjie(p2,p3,self.merge):
					d['cell_{}_{}'.format(i+1,j)].belong=d['cell_{}_{}'.format(i,j)].belong

		crop_list={}
		for i in range(len(lt_list_x)):
			for j in range(len(lt_list_y)):
				## crop_list字典以 “归属值” 为key，然后遍历所有单元格（一定要按顺序！），可以合并单元格
				crop_list['{},{}'.format(d['cell_{}_{}'.format(i,j)].belong[0], d['cell_{}_{}'.format(i,j)].belong[1])]= d['cell_{}_{}'.format(i,j)].rd
		
		self.crop_list = crop_list
		return crop_list,h_list, w_list
	
	def detnrec(self):
		self.crossingpointDetection()
		rop_list, h_list, w_list = self.cellRecognition()
		#return crop_list
		return self.crop_list, h_list, w_list

def generateExcelFile(path,filename,bboxes_loc,rec_content,crop_list,h_list,w_list):
	workbook = xlsxwriter.Workbook(os.path.join(path + '/' + filename))     # 创建新的工作簿
	worksheet = workbook.add_worksheet()   # 添加新的工作表
	# 先按行列数设置单元格，不管单元格合并格式
	col_alpha=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

	for i in range(len(w_list)):
		worksheet.set_column('{}:{}'.format(col_alpha[i],col_alpha[i]),w_list[i]/6) 
	for j in range(len(h_list)):
		worksheet.set_row(j+1,h_list[j])
		
	merge_format = workbook.add_format({
		'bold':     True,
		'border':   1,
		'align':    'left',#水平居中
		'valign':   'vcenter',#垂直居中
		#'fg_color': '#D7E4BC',#颜色填充
	})

	header_format = workbook.add_format({
		'bold':     True,
		'border':   1,
		'align':    'center',#水平居中
		'valign':   'vcenter',#垂直居中
		#'fg_color': 'blue',#颜色填充
	})

	def is_inside(cell, box):
		c1,c2,c3,c4 = cell
		b1,b2,b3,b4 = box
		if b1>c1 and b2>c2 and b3<c3 and b4<c4:
			return True
		else:
			return False
	
	tmpmax=0
	tmpmin=1e6
	zlt=[]  # 整张表格的最左上角点坐标
	zrd=[]  # 整张表格的最右下角点坐标, 只存储，没有使用
	for key in crop_list.keys():
		lt=[int(i) for i in key.split(',')]
		rd=crop_list[key]
		#cv2.imwrite('/home/elimen/Data/dbnet_pytorch/test_results_cell/{}.jpg'.format(key),raw[lt[1]:rd[1],lt[0]:rd[0]])  # 图片裁剪格式 img[y1:y2,x1:x2] or img[(y1,x1),(y2,x2)]

		if sum(rd)>tmpmax:
			zrd=rd
			tmpmax=sum(rd)
		if sum(lt)<tmpmin:
			zlt=lt
			tmpmin=sum(lt)
		
	'''
	collect and write the header first
	'''
	stored_index = []
	for key in crop_list.keys():
		lt = [int(i) for i in key.split(',')]
		rd = crop_list[key]

		for i in range(len(bboxes_loc)):
			box = bboxes_loc[i]
			cell = [lt[0],lt[1],rd[0],rd[1]]
			if is_inside(cell,box):
				stored_index.append(i)
			
	tmp_index = [ind for ind in range(len(bboxes_loc))]
	header_index = []
	for j in range(len(tmp_index)):
		if tmp_index[j] not in stored_index:
			header_index.append(j)
	if header_index:
		header = ''
		for j in range(len(header_index)):
			header += rec_content[j]+'\n'
	worksheet.set_row(0, sum(h_list)/len(h_list)*4)
	worksheet.merge_range('{}{}:{}{}'.format('A',1,chr(ord('A')+len(w_list)-1),1),'{}'.format(header),header_format)  # 合并单元格

	'''
	根据crop_list, 遍历每个单元格，然后分配行列序号
	'''
	for key in crop_list.keys():
		lt = [int(i) for i in key.split(',')]
		rd = crop_list[key]

		content = []
		for i in range(len(bboxes_loc)):
			box = bboxes_loc[i]
			cell = [lt[0],lt[1],rd[0],rd[1]]
			if is_inside(cell,box):
				content.append(rec_content[i].split('\n')[0])
	
		lt_dist2ori = [lt[0]-zlt[0],lt[1]-zlt[1]]
		rd_dist2ori = [rd[0]-zlt[0],rd[1]-zlt[1]]

		## 水平方向 
		for i in range(len(w_list)+1):
			# 左上角
			if lt_dist2ori[0]==sum(w_list[:i]):
				lt_col=chr(ord('A')+i)
				#print(lt_col)
			# 右下角
			if rd_dist2ori[0]==sum(w_list[:i]):
				rd_col=chr(ord('A')+i-1)
				#print(rd_col)
		## 竖直方向
		for i in range(len(h_list)+1):
			# 左上角
			if lt_dist2ori[1]==sum(h_list[:i]):
				lt_row=i+2
				#print(lt_row)
			# 右下角
			if rd_dist2ori[1]==sum(h_list[:i]):
				rd_row=i+1
				#print(rd_row)

		contents = ''
		if content:
			for k in range(len(content)-1):
				contents += content[k] + '\n'
			contents += content[len(content)-1]

		if lt_col==rd_col and lt_row==rd_row:
			worksheet.write('{}{}'.format(lt_col,lt_row),'{}'.format(contents),merge_format)   # 写入内容
		else:
			worksheet.merge_range('{}{}:{}{}'.format(lt_col,lt_row,rd_col,rd_row),'{}'.format(contents),merge_format)  # 合并单元格
	
	workbook.close()
	res_df = pd.read_excel(os.path.join(path + '/' + filename))
	resDict = Dict()
	headline = []
	for i in range(len(res_df.iloc[0])):
		headline.append(res_df.iloc[0,i])

	contents = []
	for i in range(1,res_df.shape[0]):
		row = []
		for j in range(res_df.shape[1]):
			row.append(res_df.iloc[i,j])
		contents.append(row)

	resDict.header = res_df.columns[0]
	resDict.data.columns = headline
	resDict.data.contents = contents

	return resDict

''' Discarded '''
class tableExtration():
	def __init__(self,srcImg,logPath):
		self.src = srcImg
		self.logPath = logPath
		self.gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
		self.binary = cv2.adaptiveThreshold(~self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
		cv2.imwrite(self.logPath+"gray.jpg",self.gray)
		# 横纵坐标数组
		self.y_point_arr = []
		self.x_point_arr = []
		# 单元格长宽集合
		self.h_list = []
		self.w_list = []

	def recognizeTable(self):
		rows, cols = self.binary.shape
		scale = 20
		scale2 = 15
		## 识别横线:
		# 自适应获取核值
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
		kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale2, 1))
		eroded_col = cv2.erode(self.binary, kernel, iterations=1)
		self.dilated_col = cv2.dilate(eroded_col, kernel2, iterations=1)
		cv2.imwrite(self.logPath+"dilated_col.jpg",self.dilated_col)

		# 识别竖线：
		# scale = 40 # scale越大，越能检测出不存在的线
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // 35)) # scale 
		kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // 20))  #scale2
		eroded_row = cv2.erode(self.binary, kernel, iterations=1)
		self.dilated_row = cv2.dilate(eroded_row, kernel2, iterations=1)

		# 横竖线交点 -- 将识别出来的横竖线合起来
		self.bitwise_and = cv2.bitwise_and(self.dilated_col, self.dilated_row)  # 对二值图进行 与操作，即可求得交点

		# 标识表格轮廓
		self.merge = cv2.add(self.dilated_col, self.dilated_row)
		ret, self.binary = cv2.threshold(self.merge, 127, 255, cv2.THRESH_BINARY)

		# 将交点标识提取出来，存放在ys，xs
		self.ys, self.xs = np.where(self.bitwise_and > 0)
		return self.ys, self.xs  

	def filterCrossPoint(self):
		# 通过排序，排除掉相近的像素点，只取相近值的最后一点
		# 这个3就是两个像素点的距离，不是固定的，根据不同的图片会有调整，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
		sort_x_point = np.sort(self.xs)
		tmpIndex = 0
		for i in range(len(sort_x_point) - 1):
			if sort_x_point[i + 1] - sort_x_point[i] > 3:
				midIndex = (tmpIndex + i) // 2
				self.x_point_arr.append(sort_x_point[midIndex])  # 要将中位数的点加入
				tmpIndex = i + 1
			i = i + 1
		self.x_point_arr.append(sort_x_point[i])  

		sort_y_point = np.sort(self.ys)
		tmpIndex = 0
		for i in range(len(sort_y_point) - 1):
			if sort_y_point[i + 1] - sort_y_point[i] > 3:
				midIndex = (tmpIndex + i) // 2
				self.y_point_arr.append(sort_y_point[midIndex])
				tmpIndex = i + 1
			i = i + 1
		self.y_point_arr.append(sort_y_point[i])

		# 单元格长宽集合
		self.h_list = [self.y_point_arr[i+1] - self.y_point_arr[i] for i in range(len(self.y_point_arr)-1)]
		self.w_list = [self.x_point_arr[i+1] - self.x_point_arr[i] for i in range(len(self.x_point_arr)-1)]

		return True

	'''
	函数 islianjie() 用于判断两点之间是否有连接。原理：两点之间 取多条 垂直的 横截线段，如果有像素值为0的，可以判断这两点是断开的。
	'''
	def islianjie(self, p1, p2, img): # 坐标p的格式是先y轴后x轴
		offset = 10  # ！！重要参数：横截线段的长度 
		if p1[0] == p2[0]:   # y坐标相同，在同一横线
			for i in range(min(p1[1],p2[1]),max(p1[1],p2[1])+1):
				if sum( [ img[j,i] for j in range( max(p1[0]-offset, 0), min(p1[0]+offset, img.shape[0]) ) ] )==0: # img mask 格式也是先y后x
					return False
			return True

		elif p1[1] == p2[1]:  # x坐标相同，在同一竖线
			for i in range(min(p1[0],p2[0]), max(p1[0],p2[0])+1):   # y轴变化范围 
				if sum( [img[i,j] for j in range(max(p1[1]-offset,0), min(p1[1]+offset,img.shape[1])) ] ) == 0:   # x轴变化范围
					return False
			return True

		else:
			return False

	# 构建一个类，作用类似于结构体
	class cell:
		def __init__(self,lt,rd,belong):
			self.lt=lt
			self.rd=rd
			self.belong=belong   # 也是以角点来表示，一个单元格归属于它的左上角，如果归属同一个点，说明是同一个单元格！

	# 通过角点检查文本框是否在单元格内
	def is_inside(self, cell, box):
		c1,c2,c3,c4 = cell
		b1,b2,b3,b4 = box
		if b1>c1 and b2>c2 and b3<c3 and b4<c4:
			return True
		else:
			return False

	def excelGeneration(self,resPath):
		'''
		关键点：2. 使用xlsxwriter库，编辑生成相应的Excel格式
		'''
		workbook = xlsxwriter.Workbook('{}{}.xlsx'.format(respath,img_name))     # 创建新的工作簿
		worksheet = workbook.add_worksheet()   # 添加新的工作表

		# 先按行列数设置单元格，不管单元格合并格式
		col_alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
		for i in range(len(self.w_list)):
			worksheet.set_column('{}:{}'.format(col_alpha[i],col_alpha[i]),w_list[i]/6) 
		for j in range(len(self.h_list)):
			worksheet.set_row(j+1, self.h_list[j])
		
		lt_list_x = x_point_arr[:-1]  # 取前面的n-1个值，最后一个不取
		lt_list_y = y_point_arr[:-1]
		rd_list_x = x_point_arr[1:]   # 从第2个值开始，第一个值不取，共n-1个值
		rd_list_y = y_point_arr[1:]
		'''
		关键中的关键： d是一个字典，每个 键值对 对应 一个单元格； value 的格式是 类cell 的实例，包含了左上角坐标，右下角坐标，以及单元格归属的左上角坐标。
		'''
		d={}
		for i in range(len(lt_list_x)):
			for j in range(len(lt_list_y)):
				d['cell_{}_{}'.format(i,j)] = cell( [lt_list_x[i],lt_list_y[j]], [rd_list_x[i],rd_list_y[j]], [lt_list_x[i],lt_list_y[j]])

		cv2.imwrite(respath+"6_checkmerge.jpg", merge)
		checkImg = cv2.cvtColor(merge,cv2.COLOR_GRAY2BGR)
		for i in range(len(lt_list_x)):
			for j in range(len(lt_list_y)):
				## p点格式为(y,x)。假设 左上角 lt(y1,x1), 右下角 rd(y2,x2) ==> 左下角 p1(y2,x1), 右上角 p3(y1,x2)
				p1 = [d['cell_{}_{}'.format(i,j)].rd[1], d['cell_{}_{}'.format(i,j)].lt[0]]  #左下点 
				p2 = [d['cell_{}_{}'.format(i,j)].rd[1], d['cell_{}_{}'.format(i,j)].rd[0]]  #右下点 
				p3 = [d['cell_{}_{}'.format(i,j)].lt[1], d['cell_{}_{}'.format(i,j)].rd[0]]  #右上点
				
				cv2.circle(checkImg,(p1[1],p1[0]),2,(0,0,255))
				cv2.circle(checkImg,(p2[1],p2[0]),2,(0,0,255))
				cv2.circle(checkImg,(p3[1],p3[0]),2,(0,0,255))
				cv2.imwrite(respath+"7_checkcorner.jpg", checkImg)
				## 查看两点之间是否连接，确定单元格归属
				if not islianjie(p1,p2,merge):
					d['cell_{}_{}'.format(i,j+1)].belong = d['cell_{}_{}'.format(i,j)].belong
				if not islianjie(p2,p3,merge):
					d['cell_{}_{}'.format(i+1,j)].belong = d['cell_{}_{}'.format(i,j)].belong

		crop_list={}
		for i in range(len(lt_list_x)):
			for j in range(len(lt_list_y)):
				## crop_list字典以 “归属值” 为key，然后遍历所有单元格（一定要按顺序！），可以合并单元格
				crop_list['{},{}'.format(d['cell_{}_{}'.format(i,j)].belong[0], d['cell_{}_{}'.format(i,j)].belong[1])]= d['cell_{}_{}'.format(i,j)].rd

		print('crop_list length:')
		print(len(crop_list))


		tmpmax=0
		tmpmin=1e6
		zlt=[]  # 整张表格的最左上角点坐标
		zrd=[]  # 整张表格的最右下角点坐标
		for key in crop_list.keys():
			lt=[int(i) for i in key.split(',')]
			rd=crop_list[key]
			#cv2.imwrite('/home/elimen/Data/dbnet_pytorch/test_results_cell/{}.jpg'.format(key),raw[lt[1]:rd[1],lt[0]:rd[0]])  # 图片裁剪格式 img[y1:y2,x1:x2] or img[(y1,x1),(y2,x2)]

			if sum(rd)>tmpmax:
				zrd=rd
				tmpmax=sum(rd)
			if sum(lt)<tmpmin:
				zlt=lt
				tmpmin=sum(lt)
			

		merge_format = workbook.add_format({
			'bold':     True,
			'border':   1,
			'align':    'left',#水平居中
			'valign':   'vcenter',#垂直居中
			#'fg_color': '#D7E4BC',#颜色填充
		})

		header_format = workbook.add_format({
			'bold':     True,
			'border':   1,
			'align':    'center',#水平居中
			'valign':   'vcenter',#垂直居中
			#'fg_color': 'blue',#颜色填充
		})

		'''
		collect and write the header first
		'''
		stored_index = []
		print(bboxes_loc)
		for key in crop_list.keys():
			lt = [int(i) for i in key.split(',')]
			rd = crop_list[key]

			for i in range(len(bboxes_loc)):
				box = bboxes_loc[i]
				cell = [lt[0],lt[1],rd[0],rd[1]]
				if is_inside(cell,box):
					stored_index.append(i)
		tmp_index = [ind for ind in range(len(bboxes_loc))]
		header_index = []
		for j in range(len(tmp_index)):
			if tmp_index[j] not in stored_index:
				header_index.append(j)
		if header_index:
			header = ''
			for j in range(len(header_index)):
				header += rec_content[j]
		worksheet.set_row(0, sum(h_list)/len(h_list)*4)
		worksheet.merge_range('{}{}:{}{}'.format('A',1,chr(ord('A')+len(w_list)-1),1),'{}'.format(header),header_format)  # 合并单元格


		'''
		根据crop_list, 遍历每个单元格，然后分配行列序号
		'''
		for key in crop_list.keys():
			lt = [int(i) for i in key.split(',')]
			rd = crop_list[key]

			content = []
			for i in range(len(bboxes_loc)):
				box = bboxes_loc[i]
				cell = [lt[0],lt[1],rd[0],rd[1]]
				if is_inside(cell,box):
					content.append(rec_content[i].split('\n')[0])

			lt_dist2ori = [lt[0]-zlt[0],lt[1]-zlt[1]]
			rd_dist2ori = [rd[0]-zlt[0],rd[1]-zlt[1]]

			## 水平方向 
			for i in range(len(w_list)+1):
				# 左上角
				if lt_dist2ori[0]==sum(w_list[:i]):
					lt_col=chr(ord('A')+i)
					#print(lt_col)
				# 右下角
				if rd_dist2ori[0]==sum(w_list[:i]):
					rd_col=chr(ord('A')+i-1)
					#print(rd_col)
			## 竖直方向
			for i in range(len(h_list)+1):
				# 左上角
				if lt_dist2ori[1]==sum(h_list[:i]):
					lt_row=i+2
					#print(lt_row)
				# 右下角
				if rd_dist2ori[1]==sum(h_list[:i]):
					rd_row=i+1
					#print(rd_row)

			contents = ''
			if content:
				for k in range(len(content)-1):
					contents += content[k] + '\n'
				contents += content[len(content)-1]

			if lt_col==rd_col and lt_row==rd_row:
				worksheet.write('{}{}'.format(lt_col,lt_row),'{}'.format(contents),merge_format)   # 写入内容
			else:
				worksheet.merge_range('{}{}:{}{}'.format(lt_col,lt_row,rd_col,rd_row),'{}'.format(contents),merge_format)  # 合并单元格

		workbook.close()
	
	def __call__(self):
		self.recognizeTable()
		self.filterCrossPoint()
		return True

	def check_internal_results(self):
		cv2.imwrite(self.logPath+"dilated_col.jpg",self.dilated_col)
		cv2.imwrite(self.logPath+"dilated_row.jpg",self.dilated_row)
		cv2.imwrite(self.logPath+"bitwise_and.jpg",self.bitwise_and)
		print(self.y_point_arr)
		print(self.x_point_arr)
		print(self.h_list)
		print(self.w_list)
		return True



if __name__ == "__main__":
	imgPath='/home/elimen/Data/dbnet_pytorch/test_images/rotated02.jpg'
	respath='/home/elimen/Data/dbnet_pytorch/test_results/tableExtration/'
	src = cv2.imread(imgPath,1)
	tab_rec = TabRecognition(src,respath)
	tab_rec.crossingpointDetection()
