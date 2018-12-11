'''
https://susanqq.github.io/UTKFace/

The labels of each face image is embedded in the file name, formated like 

[age]_[gender]_[race]_[date&time].jpg

[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
'''

import os

path = "D:/xrvision/XRV_projects/age_gender_ethicity_dataset/UTKface_dataset/noelnet_v4/dataset/UTKFace"
#folders = []
#
#for folder in folders:
#    if not os.path.exists(os.path.join(path,folder)):
#        os.mkdir(os.path.join(path,folder))
    
lst_name=[]   
for subdir, dirs, files in os.walk(path):
    if "jpg" in str(files):
        lst_name.append(files)

lst_name_new = []
for i in range(len(lst_name[0])):
    lst_name_new.append(str(lst_name[0][i]).split("_"))

#[age] is an integer from 0 to 116, indicating the age
#[gender] is either 0 (male) or 1 (female)
#[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others 
myagelist = []
for x in lst_name_new:
    myagelist.append(x[0])
ages = list(set(myagelist))
ages
mygenderlist = [0,1]     #0 (male) or 1 (female)
myracelist=[0,1,2,3,4]  #White, Black, Asian, Indian, and Others
i=1

for race in myracelist:
    for gender in mygenderlist:
        for age in ages:
            for f in sorted(lst_name_new):
                if (int(float(f[0])) == int(float(age)) ) and (int(float(f[1])) == gender) and (int(float(f[2])) == race):
                    if not os.path.exists(os.path.join(path, (str(age) + "_" + str(gender) + "_" + str(race))  )):
                        os.mkdir(os.path.join(path,(str(age) + "_" + str(gender) + "_" + str(race))  ))
                    
                    os.rename(path + '/' + ("_".join(f)) , path + '/' + (str(age) + "_" + str(gender) + "_" + str(race)) + '/image' + str(i) + ".jpg"   )
                    i=i+1
    
                #print("None")
        
