import os
import glob
# Only works in this format: "D:/xrvision/foldername")
# Converts png,jpeg,gif to jpg

# Change the 2 lines below:
os.chdir("D:/xrvision/XRV_projects/age_gender_ethicity_dataset/UTKface_dataset/noelnet_v3/dataset/above60yr_males_others")
new_filename = "above60yr_males_others"

i = 1
for old_file in glob.glob("*.jpg"): 
    new = "temp_" + str(i) + ".jpg"
    os.renames(old_file, new)
    i=i+1
    
i = 1
for old_file in glob.glob("*.bmp"): 
    new = "temp0_" + str(i) + ".jpg"
    os.renames(old_file, new)
    i=i+1
    
i = 1
for old_file in glob.glob("*.png"): 
    new = "temp1_" + str(i) + ".jpg"
    os.renames(old_file, new)
    i=i+1

i = 1
for old_file in glob.glob("*.jpeg"): 
    new = "temp2_" + str(i) + ".jpg"
    os.renames(old_file, new)
    i=i+1
    
i = 1
for old_file in glob.glob("*.gif"): 
    new = "temp3_" + str(i) + ".jpg"
    os.renames(old_file, new)
    i=i+1
    
    
i = 1
for old_file in glob.glob("*.jpg"): 
    new = new_filename + str(i) + ".jpg"
    os.renames(old_file, new)
    i=i+1
    
    