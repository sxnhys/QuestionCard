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
import sys
import os
import math


class QuestionCard(object):

    
    
    ''' 构造方法需要图像名称，[答题卡类型，配置文件名称] '''
    def __init__(self, _img, _type='1', _standard='standard.json'):
        '''
            self.img : 原图
            self.type : 答题卡种类
            self.standard : 客观题标准答案
            self.rows, self.col, self.nums : 客观题行列数和题目总数（针对每一个选择题大题）
            self.answer : 客观题填涂答案
            self.optionType : 选项类型，如'A B C D'或者'A B C D E'
            self.rotated : 倾斜校正之后的BGR图
            self.rotatedGary : 倾斜校正之后的灰度图
            self.leftPosBlock, self.rightPosBlock : 题目的左右定位块
            self.result : 返回结果
            关于86的待加入
        '''
        self.img = _img
        self.type = _type
        self.standard = _standard
        self.rows, self.col, self.nums = [], 0, []
        self.answer = dict()
        self.optionType = str()
        self.rotated = None
        self.rotatedGary = None
        self.leftPosBlock, self.rightPosBlock = None, None
        self.subjective_scores = dict()
        self.result = dict()
        
    
    ''' 加载配置文件，并初始化一些值 '''
    def load_setting(self, settingJson):
        with open(settingJson, 'r', encoding='utf8') as fp:
            setting = json.loads(fp.read())
        cardType = setting.get(self.type)
        self.col = 4   # 固定是4
        for choice in cardType['choices']:
            self.rows.append(choice['row'])
            self.nums.append(choice['num'])

        self.optionType = cardType['optionType']
        # self.answer = dict.fromkeys(range(1, self.num + 1), [])
        self.answer = {k:[] for k in range(1, sum(self.nums)+1)}

        # print(self.rows, self.nums, self.answer)

    ''' 图像预处理 '''
    def initial(self):
        global tmp_pic_dir

        if not os.path.exists('TMP_PIC'):
            os.mkdir('TMP_PIC')
        os.chdir('TMP_PIC')

        tmp_pic_dir = self.img.split('\\')[-1].split('.')[0]
        if not os.path.exists(tmp_pic_dir):
            os.mkdir(tmp_pic_dir)
        else:
            os.chdir(tmp_pic_dir)
            for file in os.listdir():
                os.remove(file)
            os.chdir('..')
        os.chdir('..')

        # 加载图片，将它转换为灰阶
        img = cv2.imread(self.img)
        # img = cv2.imdecode(np.fromfile(self.img, dtype=np.uint8), -1)
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.bitwise_not(gray)
        # 二值化，图片黑白色反转
        # h, w = img.shape[:2]
        # cv2.imwrite('test.jpg', gray[h//10:h//5, w*5//12:w*3//4])
        thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imwrite('.\\TMP_PIC\\%s\\thresh.jpg' % tmp_pic_dir, thresh)
        # 得到旋转角度
        # coords = np.column_stack(np.where(thresh > 0))
        # angle = cv2.minAreaRect(coords)[-1]
        # angle = -(90 + angle) if angle < -45 else -angle

        # print(thresh.shape[-1]*0.04)
        angle = get_angle1(thresh, thresh.shape[-1]*0.04) / np.pi * 180
        # angle = -get_angle(thresh) / np.pi * 180
        # print(angle)
        # angle = 0
        # 执行仿射变换对倾斜角度校正
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # self.rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        self.rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        self.rotatedGary = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        self.rotatedGary = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC)
        # 保存处理后的图片
        cv2.imwrite('.\\TMP_PIC\\%s\\Rotated.jpg' % tmp_pic_dir, self.rotated)
        cv2.imwrite('.\\TMP_PIC\\%s\\RotatedGary.jpg' % tmp_pic_dir, self.rotatedGary)
        
        
    ''' 找到所有题目定位块 '''
    def get_pos_block(self):

        height, width = self.rotatedGary.shape
        # print(width*0.04)
        # 截取答题卡顶部部分区域寻找试卷大定位块
        cropTop = self.rotated[:height//12, :]
        cropTopGray = self.rotatedGary[:height//12, :]
        largePosBlock = detectRect(cropTop, cropTopGray, 'CropTop', width*0.04, 150, height=20)
        # print(largePosBlock)
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
        
        # print('left: ', self.leftPosBlock, 'right: ', self.rightPosBlock)
        cv2.imwrite('.\\TMP_PIC\\%s\\marked.jpg' % tmp_pic_dir, self.rotated)
        

    ''' 获取填涂答案 '''
    # 确定一对定位块之间题目的填涂答案
    # 观察图片，以选项间空白的一半为一个单位，划分如下：
    # 每个题目（包括题号）占20个单位（题号、选项每个都是占4个单位），第二第三题之间的空白占4个单位，图片首末各占1个单位
    # 一共86个单位
    def getAnswer(self, width, questionRect, row_no, pre_num):
        answerDict = dict(zip(range(1, len(self.optionType.split())+1), self.optionType.split()))
        unit = width // 86
        # 一行题目所有可涂选项的区域范围（横向），这里一行4题，一共有16个元素
        options = [(unit*i, unit*(i+4)) for i in range(1, 86 - 1, 4) if i not in [1, 1+5*4, 1+10*4, 1+11*4, 1+16*4]]

        for qr in questionRect:
            mid_x = (qr[0] + qr[0] + qr[2]) // 2    # 填涂区域的中心横向位置 (x + (x + w)) / 2
            # print(qr[3])
            for i, op in enumerate(options):
                if op[0] + 2 < mid_x < op[1] + 5 and self.option_height < qr[3]:    # 中心位置确定选项，高度用来过滤意外的杂项
                    # if row_no == 3:
                    #     print(op[0] , mid_x , op[1] , self.option_height , qr[3])
                    # self.answer.setdefault((row_no - 1) * self.col + i // 4 + 1, []).append(answerDict.get(i % 4 + 1))
                    self.answer[pre_num + (row_no - 1) * self.col + i // 4 + 1].append(answerDict.get(i % 4 + 1))

    ''' 得到客观题答案 '''
    def get_option_question(self):
        pre_row, pre_num = 0, 0
        # 遍历每一个选择题大题
        for k, row in enumerate(self.rows):
            # 根据题目定位块和指定的题目行列数截取每一行题目，如果一对定位块上下位置相差过大则抛出异常
            for i in range(row):
                lx, ly, lw, lh = self.leftPosBlock[i + pre_row]
                rx, ry, rw, rh = self.rightPosBlock[i + pre_row]
                self.option_height = lh

                if max(ly+lh, ry+rh) - min(ly, ry) > max(lh, rh) * 3:
                    raise UserWarning('客观题定位异常（1），请检测该答题卡')

                start = self.left_base_x - (self.left_crop_width - (lx + lw))
                end = self.right_base_x + rx
                sl = slice(min(ly, ry)-10, max(ly+lh, ry+rh)+10)
                cropQuestion = self.rotated[sl, start:end]
                cropQuestionGray = self.rotatedGary[sl, start:end]

                # 所有正确填涂的答案位置 (x, y, w, h)，阈值是127
                questionRect = detectRect(cropQuestion, cropQuestionGray, 'CropQuestion%s'%(i+pre_row+1), lw, 100)
                
                self.getAnswer(cropQuestion.shape[1], questionRect, i + 1, pre_num)
            pre_row = row
            pre_num = self.nums[k]


    # ''' 得到主观题的得分 '''
    # ''' type是指选择题下面开始的主观题，还是反面一开始就是主观题 '''
    # def get_subjective_score(self, _type=0, grid=21, num=21):
    #     start_index = 0 if _type else self.row
    #     end_index = len(self.leftPosBlock)
        
    #     # print(start_index, end_index)

    #     for i in range(start_index, end_index):
    #         lx, ly, lw, lh = self.leftPosBlock[i]
    #         rx, ry, rw, rh = self.rightPosBlock[i]

    #         # print(max(ly+lh, ry+rh) - min(ly, ry), max(lh, rh))

    #         if max(ly+lh, ry+rh) - min(ly, ry) > max(lh, rh) * 3:
    #             raise UserWarning('主观题定位异常，请检测该答题卡')
    #             # self.subjective_scores.append(-1)
    #             # continue

    #         start = self.left_base_x - (self.left_crop_width - (lx + lw)) + lw // 2 
    #         end = self.right_base_x + rx - rw // 2
    #         sl = slice(ry, ry+rh)
    #         cropSubjective = self.rotated[sl, start:end]
    #         cropSubjectiveGray = self.rotatedGary[sl, start:end]

    #         h, w = cropSubjective.shape[:2]
    #         unit = w // grid
    #         # print(unit)
    #         # print('subjective{}:'.format(i+1))
    #         option_ave_gray = list()  # wait_for_select = list()
    #         for j in range(grid):
    #             s = w - unit * (j+1) + unit // 20
    #             e = w - unit * j - unit // (8 - j//2)
    #             # print([cropSubjectiveGray[j, i] for j in range(0+1, 0+h) for i in range(s+1, e)])
    #             cv2.rectangle(cropSubjective, (s, 0), (e, h), (0, 0, 255), 2)
    #             ave_gray = aveGray(cropSubjectiveGray, s, 0, e - s, h)
    #             # print(ave_gray)
    #             if ave_gray > 8:
    #                 # option_score.append(j)
    #                 option_ave_gray.append((j, ave_gray))
    #                 # if ave_gray > 35:
    #                 #     wait_for_select.append(j)
    #             else:
    #                 break

    #         # print(option_ave_gray)
    #         option_ave_gray.sort(key=lambda x: x[1], reverse=True)
    #         # print(option_ave_gray)

    #         # 灰度值前两位的选项差的的比较小的话异常：两种情况 —— 都没有打标记；同时有两个都打了标记
    #         # 这个差值很难把握啊，准确说是这个方法不太好。
    #         if option_ave_gray[0][1] - option_ave_gray[1][1] < 6:
    #             # cv2.imwrite(('.\\TMP_PIC\\%s\\CropSubjective{}Error.jpg' % tmp_pic_dir).format(i+1), cropSubjective)
    #             # raise UserWarning('主观题分数异常，请检测该答题卡')
    #             self.subjective_scores[i - start_index + 1] = -1
    #             # continue
    #         else:
    #             self.subjective_scores[i - start_index + 1] = option_ave_gray[0][0]
    #         cv2.imwrite(('.\\TMP_PIC\\%s\\CropSubjective{}.jpg' % tmp_pic_dir).format(i+1), cropSubjective)


    ''' 得到主观题的得分 '''
    ''' type是指选择题下面开始的主观题，还是反面一开始就是主观题 '''
    def get_subjective_score(self, _type=0, grid=21, num=21):
        start_index = 0 if _type else sum(self.rows)
        end_index = len(self.leftPosBlock)
        
        # print(start_index, end_index)

        for i in range(start_index, end_index):
            # print('ia', i)
            lx, ly, lw, lh = self.leftPosBlock[i]
            rx, ry, rw, rh = self.rightPosBlock[i]

            # print(max(ly+lh, ry+rh) - min(ly, ry), max(lh, rh))

            if max(ly+lh, ry+rh) - min(ly, ry) > max(lh, rh) * 3:
                raise UserWarning('主观题定位异常，请检测该答题卡')
                # self.subjective_scores.append(-1)
                # continue

            start = self.left_base_x - (self.left_crop_width - (lx + lw)) + lw*1 // 2 
            end = self.right_base_x + rx - rw*1 // 2
            # start = self.left_base_x - (self.left_crop_width - lx)
            # end = self.right_base_x + rx + rw
            sl = slice(ry-rh*4//5, ry+rh+rh)
            cropSubjective = self.rotated[sl, start:end]
            cropSubjectiveGray = self.rotatedGary[sl, start:end]

            # 旋转矫正
            cropSubjective = rotate(cropSubjective, lx+lw, ly, self.right_base_x + rx, ry)
            # 根据打勾的习惯，取下面2/3的部分检测，因为打勾的时候很容易在下一个格子的上方收笔；对于打五角星或者涂的情况也是适用的，不过尽量要求打分的时候标记偏数字下方，不要打个标记完全画到数字上方去
            cropSubjective = cropSubjective[cropSubjective.shape[0] // 3:, :]
            h, w = cropSubjective.shape[:2]
            # print(h, w)
            # unit = w // grid
            # s, e = 0, 0
            # for j in range(grid):
            # 	e = w * (j + 1) // grid
            # 	cv2.rectangle(cropSubjective, (s, 0), (e, h), (0, 0, 255), 2)
            # 	s = e + 1

            cv2.imwrite(('.\\TMP_PIC\\%s\\CropSubjective{}.jpg' % tmp_pic_dir).format(i+1), cropSubjective)

            # 检测红色
            cropSubjective = detect_red(cropSubjective)
            cv2.imwrite(('.\\TMP_PIC\\%s\\CropSubjectiveDetectRed{}.jpg' % tmp_pic_dir).format(i+1), cropSubjective)

            sum_c = 0
            scores_index = []
            # 统计255的总数，也就是红色部分的像素点数
            for row in cropSubjective:
                # sum_c += list(row).count(255)
                for col in row:
                    if sum(col) != 0:
                        sum_c += 1
            # print(sum_c)
            if sum_c < 20:   # 总数必须达到一定的数量才行，因为检测红色可能存在一些杂项，这个阈值可以调整
                self.subjective_scores[i - start_index + 1] = -1
                continue
            # 统计每个区域里面255数量，根据占总数的比例判断标记的分数
            s, e = 0, 0
            for j in range(grid):
                c = 0
                e = w * (j + 1) // grid
                for row in cropSubjective[:, s:e]:
                    # c += list(row).count(255)
                    for col in row:
                        if sum(col) != 0:
                            c += 1
                s = e + 1
                # print(sum_c, c)
                if c / sum_c > 0.3:  # 这个比例不好把握，要严格要求标记不要出界
                    scores_index.append(j)

            if len(scores_index) != 1:   # 可能存在多个分数有标记，所以不允许出现标记涂改的情况
                self.subjective_scores[i - start_index + 1] = -2
            else:
                self.subjective_scores[i - start_index + 1] = grid - scores_index[0] - 1
            # print('ib', i)

    ''' 判断是一份试卷的正面还是反面（也有可能有大于两页），并计算总得分 '''
    def get_grade(self):
        self.result = {
            'face': 0,
            'errno': 0,
            'errtype': None, 
            'errmsg': None,
        }
        try:
            self.get_pos_block()
        except Exception as e:
            raise UserWarning('试卷定位异常（1），请检测该答题卡')

        # 左右定位块数量不等，或定位块对数少于客观题的行数（后者仅针对横向填涂答题卡），抛出异常
        if len(self.leftPosBlock) != len(self.rightPosBlock):
            raise UserWarning('试卷定位异常（2），请检测该答题卡')

        h, w = self.rotatedGary.shape
        # 判断是试卷哪一页
        if self.left_base_x > w - self.right_base_x:
            if len(self.leftPosBlock) < sum(self.rows):
                raise UserWarning('客观题定位异常（2），请检测该答题卡')
            self.get_option_question()
            # print('客观题填涂答案已成功获取！')
            self.get_subjective_score()
            # print('主观题分数已成功获取！')
        else:
            self.result['face'] = 1
            self.get_subjective_score(_type=1)
            # print('主观题分数已成功获取！')


''' 计算指定区域内平均灰度 '''
def aveGray(grayimg, x, y, w, h):
    sumGray = sum(grayimg[j, i] for j in range(y+1, y+h) for i in range(x+1, x+w))
    return int(sumGray / ((w - 1) * (h - 1)) + 0.5) if (w - 1) * (h - 1) != 0 else 0


''' 在所有边缘检测的结果中找出需要的矩形部分 '''
def detectRect(img, grayimg, imgName='', width=30, threshold=200, sortkey=0, height=10):
    # 边缘检测，会有很多很多奇奇怪怪的轮廓
    image, contours, hierarchy = cv2.findContours(grayimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posBlock = list()
    # 将所有轮廓用矩形框出，大于一定宽度并且区域内灰度平均值大于阈值的，就是需要的定位块或者填涂块
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # if 'CropTop' in imgName:
        #     print(w, h, aveGray(grayimg, x, y, w, h))
        if h > height and w > width and aveGray(grayimg, x, y, w, h) > threshold:
            # if 'CropQuestion' in imgName:
            #     print(w, h, aveGray(grayimg, x, y, w, h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            posBlock.append((x, y, w, h))
    # 保存相应的灰度图像和带矩形标记的BGR图
    cv2.imwrite(('.\\TMP_PIC\\%s\\{}.jpg' % tmp_pic_dir).format(imgName), grayimg)
    cv2.imwrite(('.\\TMP_PIC\\%s\\contours{}.jpg' % tmp_pic_dir).format(imgName), img)
    return sorted(posBlock, key=lambda pb: pb[sortkey])


''' 根据大定位块确定需要选择的角度，找大定位块的过程与之后的过程重复（暂时如此）'''
def get_angle1(grayimg, width=60, threshold=200):
    # 边缘检测，会有很多很多奇奇怪怪的轮廓
    image, contours, hierarchy = cv2.findContours(grayimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largePosBlock = list()
    # 将所有轮廓用矩形框出，大于一定宽度并且区域内灰度平均值大于阈值的，就是需要的定位块
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 15 and w > width and aveGray(grayimg, x, y, w, h) > threshold:
            largePosBlock.append((x, y, w, h))

    # print(len(largePosBlock))
    if len(largePosBlock) != 2:
        raise UserWarning('试卷扫描异常，未定位到试卷')

    largePosBlock.sort(key=lambda x: x[0])
    delta_x = largePosBlock[1][0] - largePosBlock[0][0]
    delta_y = largePosBlock[1][1] - largePosBlock[0][1]
    return math.atan(delta_y / delta_x)


''' 由于试卷整体旋转之后主观题部分有的还是歪的明显，所以这里对每个主观题部分截图旋转，方法同上 '''
def rotate(img, *xy2):
    lx, ly, rx, ry = xy2
    delta_x = rx - lx
    delta_y = ry - ly
    angle = math.atan(delta_y / delta_x) / np.pi * 180

    h, w = img.shape[:2]
    # 以最右边线中点为旋转中心
    center = (w, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


''' 主观题部分检测红色 '''
def detect_red(img):
    # 红色在HSV模型中有两个取值范围, lower里 s,v 由原先的 43, 46 改成了现在的80，可以排除一些原先的杂项
    lower_red1 = np.array([0,80,80])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([156,80,80])
    upper_red2 = np.array([180,255,255])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    # h, w = mask.shape[:2]

    # detect red
    res = cv2.bitwise_and(img, img, mask=mask2)

    # for row in res:
    #     for col in row:
    #     if 0 not in col:
    # 			print(col)

    return res#mask


def QC_API(pic_path=-1, setting='setting.json', *, _type):

    if pic_path == -1:
            raise UserWarning('请提供图片路径')
    qc = QuestionCard(pic_path, _type)
    
    try:
        # 预处理图片，加载配置文件
        qc.initial()
        qc.load_setting(setting)
        # 获取客观题答案和主观题分数
        qc.get_grade()

        qc.result['obj'] = qc.answer if qc.result['face'] == 0 else None
        qc.result['subj'] = qc.subjective_scores

    except Exception as e:
        qc.result['errno'] = 1
        qc.result['errtype'] = type(e).__name__
        qc.result['errmsg'] = str(e)
        qc.result['subj'] = qc.subjective_scores
        qc.result['obj'] = qc.answer if qc.result['face'] == 0 else None

    return qc.result


if __name__ == '__main__':
    qc = QuestionCard('pic', '4')
    qc.load_setting('setting.json')