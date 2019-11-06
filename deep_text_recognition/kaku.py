import os, glob, cv2
import numpy as np
import torch
import torchvision.transforms as transforms

te = glob.glob('./demo_images/*.png')

nup_img_list = []
resize_img_list = []
for index in range(len(te)):
    imgs = cv2.imread(te[index])
    resize_img = cv2.resize(imgs, (32,32))
    resize_img_list.append(resize_img)
    print(te[index])
    #nup_img = np.array(resize_img)
    #print(type(nup_img))
    nup_img_list.append(resize_img)

#nup_img_list = np.array(nup_img_list)
#tensor_img = torch.from_numpy(nup_img_list)
toTensor = transforms.ToTensor()
tensor_img = [toTensor(img) for img in resize_img_list]
image_tensors = torch.cat([t.unsqueeze(0) for t in tensor_img], 0)
print(type(tensor_img))
print(image_tensors.shape)