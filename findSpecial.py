import re

files = [
	'test_res(biological_7).txt',
	'test_res(biological_8).txt',
	'test_res(chemistry_7).txt',
	'test_res(chemistry_8).txt',
	'test_res(physical_7).txt',
	'test_res(physical_8).txt',
	'test_res(test_pic).txt',
]


with open('special.txt', 'w', encoding='utf8') as sf:
    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            subj = re.findall(r'\((.*)\)', file)[0]
            s = f.read()
            sf.write('=' * 20 + subj + ' -1' + '=' * 20 + '\n')
            l = re.findall(r'\\([0-9_]*)\.jpg\n(.*)-1', s)
            ll = [i[0]+'\n' for i in l]
            sf.writelines(ll)

            sf.write('=' * 20 + subj + ' -2' + '=' * 20 + '\n')
            l = re.findall(r'\\([0-9_]*)\.jpg\n(.*)-2', s)
            ll = [i[0]+'\n' for i in l]
            sf.writelines(ll)
