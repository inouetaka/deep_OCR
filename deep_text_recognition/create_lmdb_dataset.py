# -*- coding: utf-8 -*-
""" CRNNtorchリポジトリの修正バージョン https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    トレーニングと評価用のLMDBデータセットを作成します。
     ARGS：
         inputPath：imagePathを開始する入力フォルダーパス
         outputPath：LMDB出力パス
         gtFile：画像パスとラベルのリスト
         checkValid：trueの場合、すべての画像の有効性を確認します
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()
    error_count = 0
    nSamples = len(datalist)
    for i in range(nSamples):
        if "\t" in datalist[i]:
            print(datalist[i].strip('\n').split('\t'))
            imagePath, label = datalist[i].strip('\n').split('\t')
            imagePath = os.path.join(inputPath, imagePath)
        else:
            print("例外処理", datalist[i])
            error_count += 1

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s が存在しません。' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s 有効な画像ではありません' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if cnt % 10000 == 0:
            writeCache(env, cache)
            cache = {}
            print('書き込み中 %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('%d 個のサンプルでデータセットが作成されました。' % nSamples)
    print(f"エラー数:{error_count}/{nSamples} [{(error_count/nSamples)*100}%]")


if __name__ == '__main__':
    fire.Fire(createDataset)
