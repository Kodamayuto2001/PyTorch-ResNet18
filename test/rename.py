import os 
import shutil

def rename(oldDir,newDir="dataset/renamed/"):
    try:
        os.makedirs(newDir)
    except FileExistsError:
        pass

    cnt = 0
    for f in os.listdir(oldDir):
        print(cnt)
        shutil.copyfile(oldDir+f,newDir+str(cnt)+".jpg")
        cnt += 1

rename("dataset/all/")