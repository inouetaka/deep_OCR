# -*- coding: utf-8 -*-

import os
import argparse
import unicodedata


def word_normalize(opt):
    os.makedirs(opt.output_path, exist_ok=True)

    with open(opt.input_path, "r") as rf:
        gt_words = rf.readlines()

    path_list = []
    gt_list = []
    for i in range(len(gt_words)):
        path, gt = gt_words[i].split("\t")
        gt, _ = gt.split("\n")
        path_list.append(path)
        gt_list.append(gt)

    print('split data Consistency of size => origin_size = split_data_size? : {}'.format(len(gt_words) == len(gt_list)))

    norm_words = []
    for i in range(len(gt_list)):
        norm_word = unicodedata.normalize("NFKC", gt_list[i])
        norm_words.append(norm_word)
        if gt_list[i] is not norm_word:
            print("BEFORE: {}".format(gt_list[i]))
            print("AFTER : {}".format(norm_word))
            print("CONSISTENCY: {}".format(str(gt_list[i] == norm_word)))
            print("*-"*30)
    print('Consistency of size => origin_size = normalize_data_size?: {}'.format(len(gt_words) == len(norm_words)))

    with open(f"{opt.output_path}/{opt.output_file_name}", "w") as wf:
        for i in range(len(path_list)):
            wf.write("{}\t{}\n".format(path_list[i], norm_words[i]))

    with open(f"{opt.output_path}/{opt.output_file_name}", "r") as r:
        conf_len = r.readlines()
    print('Consistency of size => origin_size = normalize_dataset_size?: {}'.format(len(gt_words) == len(conf_len)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True, help='path to input file')
    parser.add_argument('--output-path', required=True, help='path to output file')
    parser.add_argument('--output-file-name', required=True, help='output file name')

    opt = parser.parse_args()

    word_normalize(opt)