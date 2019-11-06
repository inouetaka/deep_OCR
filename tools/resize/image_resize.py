import cv2
import os
import glob
import numpy as np
import argparse


def data_loader(path):
    img = cv2.imread(path)
    h, w, _ = img.shape
    mu_b = img.T[0].flatten().mean()
    mu_g = img.T[1].flatten().mean()
    mu_r = img.T[2].flatten().mean()
    return img, h, w, _, mu_b, mu_g, mu_r


def gen_resize_img(path, opt):
    img, h, w, _, mu_b, mu_g, mu_r = data_loader(path)
    resize_img = cv2.resize(img, (opt.resize_width, int(opt.resize_width * h / w)))
    r_h, r_w, r_c = resize_img.shape
    if r_h <= opt.resize_height:
        H_PAD = opt.resize_height - r_h
        b = np.full((H_PAD, r_w), int(mu_b))
        g = np.full((H_PAD, r_w), int(mu_g))
        r = np.full((H_PAD, r_w), int(mu_r))
        bgr = np.stack([b, g, r], 2)
        new_image = np.concatenate([resize_img, bgr], 0)
        print('before:{} => middle:{} => after:{}'.format(img.shape, resize_img.shape, new_image.shape))
    elif r_h > opt.resize_height:
        w_resize_img = cv2.resize(resize_img, (int(opt.resize_height * w / h), opt.resize_height))
        r_h, r_w, r_c = w_resize_img.shape
        W_PAD = opt.resize_width - r_w
        b = np.full((r_h, W_PAD), int(mu_b))
        g = np.full((r_h, W_PAD), int(mu_g))
        r = np.full((r_h, W_PAD), int(mu_r))
        bgr = np.stack([b, g, r], 2)
        new_image = np.concatenate([w_resize_img, bgr], 1)
        print('before:{} => middle:{} => after:{}'.format(img.shape, resize_img.shape, new_image.shape))

    else:
        print(resize_img.shape)
        print("どっちにも引っかかってないよ")

    return new_image


def main(opt):
    os.makedirs(opt.output_path, exist_ok=True)
    image_path = glob.glob(opt.input_path + "/*.png")
    for p in image_path:
        resized_new_image = gen_resize_img(p, opt)
        image_name = p.split('/')
        cv2.imwrite(opt.output_path + image_name[len(image_name)-1], resized_new_image)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='path to input image')
    parser.add_argument('--output_path', required=True, help='path to output image')
    parser.add_argument('--resize_width', default=100, help='width you want to resize')
    parser.add_argument('--resize_height', default=32, help='height you want to resize')

    opt = parser.parse_args()

    main(opt)