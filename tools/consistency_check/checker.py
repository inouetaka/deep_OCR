import os
import shutil
import pandas as pd

data_path = "./false_checker.csv"
img_path = "../dataset/1015_DTR/"
extraction_path = './valid_img'


def Extraction(data_path, img_path, extraction_path):
    data = pd.read_csv(data_path)
    idx = os.listdir(img_path)
    os.makedirs(extraction_path, exist_ok=True)
    for valid_img in data['path']:
        if valid_img in idx:
            print('{}をい')
            shutil.move(img_path + valid_img, extraction_path)

#Extraction(data_path, img_path)
