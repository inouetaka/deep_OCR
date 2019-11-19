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
        H_PAD = int((opt.resize_height - r_h) / 2)
        if (opt.resize_height - r_h) % 2 == 0:
            b, g, r = np.full((H_PAD, r_w), int(mu_b)), np.full((H_PAD, r_w), int(mu_g)), np.full((H_PAD, r_w), int(mu_r))
            bgr = np.stack([b, g, r], 2)
            new_image = np.concatenate([bgr, resize_img, bgr], 0)
            print('before:{} => middle:{} => after:{}'.format(img.shape, resize_img.shape, new_image.shape))
        else:
            b, g, r = np.full((H_PAD, r_w), int(mu_b)), np.full((H_PAD, r_w), int(mu_g)), np.full((H_PAD, r_w), int(mu_r))
            b_, g_, r_ = np.full((1, r_w), int(mu_b)), np.full((1, r_w), int(mu_g)), np.full((1, r_w), int(mu_r))
            bgr = np.stack([b, g, r], 2)
            bgr_ = np.stack([b_, g_, r_], 2)
            new_image = np.concatenate([bgr, resize_img, bgr], 0)
            new_image = np.concatenate([bgr_, new_image], 0)
            print('before:{} => middle:{} => after:{}'.format(img.shape, resize_img.shape, new_image.shape))
    elif r_h > opt.resize_height:
        w_resize_img = cv2.resize(resize_img, (int(opt.resize_height * w / h), opt.resize_height))
        r_h, r_w, _ = w_resize_img.shape
        W_PAD = int((opt.resize_width - r_w) / 2)
        if (opt.resize_width - r_w) % 2 == 0:
            b, g, r = np.full((r_h, W_PAD), int(mu_b)), np.full((r_h, W_PAD), int(mu_g)), np.full((r_h, W_PAD), int(mu_r))
            bgr = np.stack([b, g, r], 2)
            new_image = np.concatenate([bgr, w_resize_img, bgr], 1)
            cv2.imwrite('new_image.png', new_image)
            print('before:{} => middle:{} => after:{}'.format(img.shape, resize_img.shape, new_image.shape))
        else:
            b, g, r = np.full((r_h, W_PAD), int(mu_b)), np.full((r_h, W_PAD), int(mu_g)), np.full((r_h, W_PAD), int(mu_r))
            bgr = np.stack([b, g, r], 2)
            b_, g_, r_ = np.full((r_h, 1), int(mu_b)), np.full((r_h, 1), int(mu_g)), np.full((r_h, 1), int(mu_r))
            bgr_ = np.stack([b_, g_, r_], 2)
            new_image = np.concatenate([bgr, w_resize_img, bgr], 1)
            new_image = np.concatenate([bgr_ , new_image], 1)
            cv2.imwrite('new_image.png', new_image)
            print('before:{} => middle:{} => after:{}'.format(img.shape, resize_img.shape, new_image.shape))
    else:
        print(f"漏れてるよ:{path}")
        print('before:{} => middle:{} '.format(img.shape, resize_img.shape))

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
    parser.add_argument('--resize_width', default=100, type=int, help='width you want to resize')
    parser.add_argument('--resize_height', default=32, type=int, help='height you want to resize')

    opt = parser.parse_args()

    main(opt)



