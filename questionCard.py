"""
答题卡自动识别评分
答题卡有很多类型，如客观题横向填涂和纵向填涂，还有多选题，或者选项不止是ABCD
这里先实现了横向填涂的ABCD单选题

基本步骤：
1、加载原图，先转灰度图，再二值化
2、仿射变换对倾斜角度校正
3、截取图像顶部部分，确定大的定位块
4、根据大定位块，截取答题卡左右两侧寻找试卷题目定位块
5、客观题：根据每行题目的一对定位块，以及配置文件中关于客观题行列数，进行每行题目的截取
		 识别出每行题目中被正确填涂的答案区域（设置阈值，默认127），根据比例划分确定填涂的具体是哪个题目的哪个选项

每个截图都有保存
"""


import numpy as np
import cv2
import json


class QuestionCard(object):
	
	''' 构造方法需要图像名称，[答题卡类型，配置文件名称] '''
	def __init__(self, _img, _type='1', _standard='standard.json'):
		'''
			self.img : 原图
			self.type : 答题卡种类
			self.standard : 客观题标准答案
			self.row, self.col, self.num : 客观题行列数和题目总数
			self.answer : 客观题填涂答案
			self.optionType : 选项类型，如'A B C D'或者'A B C D E'
			self.rotated : 倾斜校正之后的BGR图
			self.rotatedGary : 倾斜校正之后的灰度图
			self.leftPosBlock, self.rightPosBlock : 题目的左右定位块
			关于86的待加入
		'''
		self.img = _img
		self.type = _type
		self.standard = _standard
		self.row, self.col, self.num = 0, 0, 0
		self.answer = dict()
		self.optionType = str()
		self.rotated = None
		self.rotatedGary = None
		self.leftPosBlock, self.rightPosBlock = None, None
		self.subjective_scores = list()
	
	''' 加载配置文件，并初始化一些值 '''
	def load_setting(self, settingJson):
		with open(settingJson) as fp:
			setting = json.loads(fp.read())
		cardType = setting.get(self.type)
		self.row, self.col, self.num = cardType['row'], cardType['col'], cardType['num']
		self.optionType = cardType['optionType']
		# self.answer = dict.fromkeys(range(1, self.num + 1))

	''' 图像预处理 '''
	def initial(self):
		# 加载图片，将它转换为灰阶
		img = cv2.imread(self.img)
		print(img.shape)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		self.gray = cv2.bitwise_not(gray)
		# 二值化，图片黑白色反转
		# h, w = img.shape[:2]
		# cv2.imwrite('test.jpg', gray[h//10:h//5, w*5//12:w*3//4])
		thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		# 得到旋转角度
		# coords = np.column_stack(np.where(thresh > 0))
		# angle = cv2.minAreaRect(coords)[-1]
		# angle = -(90 + angle) if angle < -45 else -angle

		angle = -get_angle(thresh) / np.pi * 180
		print(angle)
		# angle = 0
		# 执行仿射变换对倾斜角度校正
		h, w = img.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		self.rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
		self.rotatedGary = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
		# 保存处理后的图片
		cv2.imwrite('Rotated.jpg', self.rotated)
		cv2.imwrite('RotatedGary.jpg', self.rotatedGary)

	''' 找到所有题目定位块 '''
	def get_pos_block(self):
		height, width = self.rotatedGary.shape
		# 截取答题卡顶部部分区域寻找试卷大定位块
		cropTop = self.rotated[:height//10, :]
		cropTopGray = self.rotatedGary[:height//10, :]
		largePosBlock = detectRect(cropTop, cropTopGray, 'CropTop', width*0.05, 150)
		# 根据大定位块，截取答题卡左右两侧寻找试卷题目定位块, 必须len(largePosBlock) == 2
		self.left_crop_width = 100
		self.left_base_x = largePosBlock[0][0]
		sl = slice(self.left_base_x - self.left_crop_width, self.left_base_x)
		cropLeft = self.rotated[:, sl]
		cropLeftGray = self.rotatedGary[:, sl]
		self.leftPosBlock = detectRect(cropLeft, cropLeftGray, 'CropLeft', width*0.01, 150, 1)

		self.right_base_x = largePosBlock[1][0] + largePosBlock[1][2]
		cropRight = self.rotated[:, self.right_base_x : ]
		cropRightGray = self.rotatedGary[:, self.right_base_x : ]
		self.rightPosBlock = detectRect(cropRight, cropRightGray, 'CropRight', width*0.01, 150, 1)

	''' 获取填涂答案 '''
	# 确定一对定位块之间题目的填涂答案
	# 观察图片，以选项间空白的一半为一个单位，划分如下：
	# 每个题目（包括题号）占20个单位（题号、选项每个都是占4个单位），第二第三题之间的空白占4个单位，图片首末各占1个单位
	# 一共86个单位
	def getAnswer(self, width, questionRect, row_no):
		answerDict = dict(zip(range(1, 5), self.optionType.split()))
		unit = width // 86
		# 一行题目所有可涂选项的区域范围（横向），这里一行4题，一共有16个元素
		options = [(unit*i, unit*(i+4)) for i in range(1, 86 - 1, 4) if i not in [1, 1+5*4, 1+10*4, 1+11*4, 1+16*4]]

		for qr in questionRect:
			mid_x = (qr[0] + qr[0] + qr[2]) // 2    # 填涂区域的中心横向位置 (x + (x + w)) / 2
			# print(qr[3])
			for i, op in enumerate(options):
				# if row_no == 5:
				# 	print(op[0] , mid_x , op[1] , self.option_height , qr[3])
				if op[0] < mid_x < op[1] + 5 and self.option_height < qr[3]:    # 中心位置确定选项，高度用来过滤意外的杂项
					self.answer.setdefault((row_no - 1) * self.col + i // 4 + 1, []).append(answerDict.get(i % 4 + 1))

	''' 得到客观题答案 '''
	def get_option_question(self):
		
		# 根据题目定位块和指定的题目行列数截取每一行题目，如果一对定位块上下位置相差过大则抛出异常
		for i in range(self.row):
			lx, ly, lw, lh = self.leftPosBlock[i]
			rx, ry, rw, rh = self.rightPosBlock[i]
			self.option_height = lh

			if max(ly+lh, ry+rh) - min(ly, ry) > max(lh, rh) + 30:
				raise UserWarning('定位异常，请检测该答题卡')

			start = self.left_base_x - (self.left_crop_width - (lx + lw))
			end = self.right_base_x + rx
			sl = slice(min(ly, ry)-10, max(ly+lh, ry+rh)+10)
			cropQuestion = self.rotated[sl, start:end]
			cropQuestionGray = self.rotatedGary[sl, start:end]

			# 所有正确填涂的答案位置 (x, y, w, h)，阈值是127
			questionRect = detectRect(cropQuestion, cropQuestionGray, 'CropQuestion%s'%(i+1), lw, 100)
			
			self.getAnswer(cropQuestion.shape[1], questionRect, i + 1)

		# return self.answer

	''' 得到主观题的得分 '''
	''' type是指选择题下面开始的主观题，还是反面一开始就是主观题 '''
	def get_subjective_score(self, _type=0, grid=21, num=21):
		start_index = 0 if _type else self.row
		end_index = len(self.leftPosBlock)
		
		for i in range(start_index, end_index):
			lx, ly, lw, lh = self.leftPosBlock[i]
			rx, ry, rw, rh = self.rightPosBlock[i]

			if max(ly+lh, ry+rh) - min(ly, ry) > max(lh, rh) + 30:
				raise UserWarning('定位异常，请检测该答题卡')

			start = self.left_base_x - (self.left_crop_width - (lx + lw)) + lw // 2 
			end = self.right_base_x + rx - rw // 2
			sl = slice(ry, ry+rh)
			cropSubjective = self.rotated[sl, start:end]
			cropSubjectiveGray = self.rotatedGary[sl, start:end]

			h, w = cropSubjective.shape[:2]
			unit = w // grid
			# print('subjective{}:'.format(i+1))
			option_score, wait_for_select = list(), list()
			for j in range(grid):
				s = w - unit * (j+1) + 5
				e = w - unit * j - 15
				# print([cropSubjectiveGray[j, i] for j in range(0+1, 0+h) for i in range(s+1, e)])
				cv2.rectangle(cropSubjective, (s, 0), (e, h), (0, 0, 255), 2)
				ave_gray = aveGray(cropSubjectiveGray, s, 0, e - s, h)
				if ave_gray > 10:
					option_score.append(j)
					if ave_gray > 40:
						wait_for_select.append(j)
				else:
					break
			
			if len(wait_for_select) != 1:
				raise UserWarning('主观题分数异常，请检测该答题卡')
			self.subjective_scores.append(wait_for_select[0])
			cv2.imwrite('.\\pic\\CropSubjective{}.jpg'.format(i+1), cropSubjective)

		# return self.subjective_scores

	''' 判断是一份试卷的正面还是反面（也有可能有大于两页），并计算总得分 '''
	def get_grade(self):
		self.get_pos_block()
		# 左右定位块数量不等，或定位块对数少于客观题的行数（后者仅针对横向填涂答题卡），抛出异常
		if len(self.leftPosBlock) != len(self.rightPosBlock):
			raise UserWarning('定位异常，请检测该答题卡')
		h, w = self.rotatedGary.shape
		# 判断是试卷哪一页
		if self.left_base_x > w - self.right_base_x:
			if len(self.leftPosBlock) < self.row:
				raise UserWarning('定位异常，请检测该答题卡')
			self.get_option_question()
			print('客观题填涂答案已成功获取！')
			self.get_subjective_score()
			print('主观题分数已成功获取！')
		else:
			self.get_subjective_score(_type=1)
			print('主观题分数已成功获取！')


''' 计算指定区域内平均灰度 '''
def aveGray(grayimg, x, y, w, h):
	sumGray = sum(grayimg[j, i] for j in range(y+1, y+h) for i in range(x+1, x+w))
	return sumGray // ((w - 1) * (h - 1)) if (w - 1) * (h - 1) != 0 else 0


''' 在所有边缘检测的结果中找出需要的矩形部分 '''
def detectRect(img, grayimg, imgName='', width=30, threshold=200, sortkey=0):
	# 边缘检测，会有很多很多奇奇怪怪的轮廓
	image, contours, hierarchy = cv2.findContours(grayimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	posBlock = list()
	# 将所有轮廓用矩形框出，大于一定宽度并且区域内灰度平均值大于阈值的，就是需要的定位块或者填涂块
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
		# if 'CropQuestion' in imgName:
		# 	print(w, h, aveGray(grayimg, x, y, w, h))
		if h > 10 and w > width and aveGray(grayimg, x, y, w, h) > threshold:
			if 'CropQuestion' in imgName:
				print(w, h, aveGray(grayimg, x, y, w, h))
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
			posBlock.append((x, y, w, h))
	# 保存相应的灰度图像和带矩形标记的BGR图
	cv2.imwrite('.\\pic\\{}.jpg'.format(imgName), grayimg)
	cv2.imwrite('.\\pic\\contours{}.jpg'.format(imgName), img)
	return sorted(posBlock, key=lambda pb: pb[sortkey])


def get_lines(bimg, threshold=700):
    lines = cv2.HoughLines(bimg, 1, np.pi / 180, threshold)
    lines = lines.reshape((len(lines), 2))
    thetas = lines[:, 1]
    phis = np.pi / 2 - thetas
    idx = (phis > - np.pi / 4) & (phis < np.pi / 4)
    
    return lines[idx]
    
def get_angle(bimg, bias=0):
    
    lines = get_lines(bimg)
    thetas = lines[:, 1]
    phis = np.pi / 2 - thetas

    bins = np.linspace(- np.pi /4, np.pi / 4, 451)
    freq, _ = np.histogram(phis, bins)

    idx = np.argmax(freq)
    angle = bins[idx] + bins[1] - bins[0]

    return angle + bias


if __name__ == '__main__':
	# qc = QuestionCard('initial.jpg')
	# qc = QuestionCard('test_pic\\20180511170804_001.jpg')
	# qc = QuestionCard('..\\QuestionCardPicture\\physical_8\\20180530161530_003.jpg')
	qc = QuestionCard('..\\QuestionCardPicture\\chemistry_7\\20180530162931_001.jpg')
	# qc = QuestionCard('..\\QuestionCardPicture\\biological_8\\20180530155554.jpg', _type='2')
	# 预处理图片，加载配置文件
	qc.initial()
	qc.load_setting('setting.json')
	# 获取客观题答案和主观题分数
	qc.get_grade()
	print(qc.answer, qc.subjective_scores, sep='\n')


