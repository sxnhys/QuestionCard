from questionCard import QC_API
import os, json

if __name__ == '__main__':
	result = QC_API('.\\initial.jpg', _type='1')
	# result = QC_API('F:\\QuestionCardPicture\\biological_7\\20180530160022.jpg', _type='2')
	# result = QC_API('.\\test_pic\\20180511170813.jpg', _type='1')
	# result = QC_API('F:\\QuestionCardPicture\\others\\dili\\~tmp~winscan_to_pdf_1~2018-07-16_17-40-51.jpg', _type='2')
	print(result)

	# 成功率 98%
	# base_dir = 'F:\\QuestionCardPicture\\biological_7\\'

	# 成功率 93%
	# base_dir = 'F:\\QuestionCardPicture\\biological_8\\'

	# 成功率 97.5%
	# base_dir = 'F:\\QuestionCardPicture\\chemistry_7\\'

	# 成功率 95%
	# base_dir = 'F:\\QuestionCardPicture\\chemistry_8\\'

	# 成功率 95%
	# base_dir = 'F:\\QuestionCardPicture\\physical_7\\'

	# 成功率 99.5%
	# base_dir = 'F:\\QuestionCardPicture\\physical_8\\'

	# 成功率 100%
	# base_dir = '.\\test_pic\\'


	# base_dirs = {
	# 	'F:\\QuestionCardPicture\\biological_7\\': '2', 
	# 	'F:\\QuestionCardPicture\\biological_8\\': '2',
	# 	'F:\\QuestionCardPicture\\chemistry_7\\': '1',
	# 	'F:\\QuestionCardPicture\\chemistry_8\\': '1',
	# 	'F:\\QuestionCardPicture\\physical_7\\': '1',
	# 	'F:\\QuestionCardPicture\\physical_8\\': '1',
	# 	'.\\test_pic\\': '1',
	# }

	# for base_dir in base_dirs:
	# 	print('='*15, base_dir, '='*15)
	# 	test_pics = [base_dir + pic for pic in os.listdir(base_dir)]

	# 	with open('test_res(%s).txt' % base_dir.split('\\')[-2], 'w', encoding='utf8') as f:
	# 		for i, pic in enumerate(test_pics, 1):
	# 			print('Testing%d %s' % (i, pic))
	# 			f.write('Test%d - %s' % (i, pic) + '\n')
	# 			f.write(json.dumps(QC_API(pic, _type=base_dirs[base_dir])).encode('utf-8').decode('unicode_escape') + '\n\n')