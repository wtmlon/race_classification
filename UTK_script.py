import os, shutil
list_dir = os.listdir("F:\\Race_Classification\\UTKFace")
list_0 = []
list_1 = []
list_2 = []
list_3 = []
list_4 = []
for l in list_dir:
	res = l.split('_')
	if len(res) <= 3:
		continue
	if res[2] == '0':
		list_0.append(l)
	if res[2] == '1':
		list_1.append(l)
	if res[2] == '2':
		list_2.append(l)
	if res[2] == '3':
		list_3.append(l)
	if res[2] == '4':
		list_4.append(l)

for l in list_0:
	shutil.copy("F:\\Race_Classification\\UTKFace" + '\\' + l, "F:\\Race_Classification\\UTKFace\\0")
for l in list_1:
	shutil.copy("F:\\Race_Classification\\UTKFace" + '\\' + l, "F:\\Race_Classification\\UTKFace\\1")

for l in list_2:
	shutil.copy("F:\\Race_Classification\\UTKFace"+ '\\' + l, "F:\\Race_Classification\\UTKFace\\2")

for l in list_3:
	shutil.copy("F:\\Race_Classification\\UTKFace" + '\\'  + l, "F:\\Race_Classification\\UTKFace\\3")

for l in list_4:
	shutil.copy("F:\\Race_Classification\\UTKFace" + '\\'  + l, "F:\\Race_Classification\\UTKFace\\4")

