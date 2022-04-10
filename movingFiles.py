import os
 
benign_dest = '/home/CAMPUS/smsc2020/courses/152/FinalProject/train_data/benign'
malig_dest = '/home/CAMPUS/smsc2020/courses/152/FinalProject/train_data/malignant'

source = "/home/CAMPUS/smsc2020/courses/152/FinalProject/fold5/train/"
magnifications = ["40X", "100X", "200X", "400X"]

for m in magnifications:
    currentdir = os.path.join(source, m)
    allfiles = os.listdir(currentdir)
 
    for f in allfiles:
        classification = f.split("_")[1]

        if classification == "B":
            os.rename(os.path.join(currentdir, f), os.path.join(benign_dest, f))
        elif classification == "M":
            os.rename(os.path.join(currentdir, f), os.path.join(malig_dest, f))
        else:
            print(classification)