import argparse
import time
import cv2
import glob
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import os

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device ="cpu"

def loader(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    batch_size = 10
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

    model.eval()
    return model, converter, length_for_pred, text_for_pred


# ------------------------------------------------------------------------------------------------------------------ #

def original_demo(model, converter, length_for_pred, text_for_pred, opt):
    start_time = time.time()

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    get_data = time.time()-start_time

    # predict
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # 最大長予測用
            #torch.cuda.synchronize(device)
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred).log_softmax(2)
                # 最大確率を選択し、インデックスを文字にデコードします
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # 最大確率を選択し、インデックスを文字にデコードします
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            print('-' * 80)
            print('image_path\tpredicted_labels')
            print('-' * 80)
            for img_name, pred in zip(image_path_list, preds_str):
                if 'Attn' in opt.Prediction:
                    pred = pred[:pred.find('[s]')]  # 文の終わりトークン（[s]）の後の剪定

                print(f'{img_name}\t{pred}')

        forward_time = time.time() - start_time
        print('*' * 80)
        print('get_dta_time:{:.5f}[sec]'.format(get_data))
        print('only_infer_time:{:.5f}[sec]'.format(forward_time - get_data))
        print('total_time:{:.5f}[sec]'.format(forward_time))
        print('*' * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        #opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        with open("japanese_word/japanese_word.txt", "r")as ja:
            opt.character = ja.read()
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    model, converter, length_for_pred, text_for_pred = loader(opt)
    for i in range(5):
        original_demo(model, converter, length_for_pred, text_for_pred, opt)