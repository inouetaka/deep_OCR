import h5py, os, cv2
import numpy as np


def main(input_path, output_path, split_per, data_split):
    db = h5py.File(input_path)['data']
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/train', exist_ok=True)
    os.makedirs(f'{output_path}/test', exist_ok=True)
    os.makedirs(f'{output_path}/valid', exist_ok=True)
    label_list = []
    image_list = []
    for o, k in enumerate(db.keys(), len(range(100))):
        print(o)
        d = db[k]
        print('*-' * 45)
        print(f'extracting {k}...')
        cv2.imwrite(f'{output_path}/{k}.png', d[:][..., [2, 1, 0]])
        if 'count' in d.attrs:
            c = d.attrs['count']
            for i in range(c):
                print(f'instance {i}')
                print('character bounding boxes (2 x 4 x N):')
                #print(d.attrs[f'charBB{i}'])
                print('text:')
                print(''.join([x.decode('utf8') for x in d.attrs[f'txt{i}']]))
                label_list.append(''.join([x.decode('utf8') for x in d.attrs[f'txt{i}']]))
                image_list.append(k)
        else:
            print('character bounding boxes (2 x 4 x N):')
            #print(d.attrs['charBB'])
            print('text:')
            print(''.join([x.decode('utf8') for x in d.attrs['txt']]))
            label_list.append(''.join([x.decode('utf8') for x in d.attrs['txt']]))
            image_list.append(k)
        print('label:')
        print(['train', 'val', 'test'][d.attrs['label']])

    num = len(label_list)
    split_per = [float(x.strip()) for x in split_per.split(',')]
    per = [int(len(label_list) * n) for n in [split_per[0], split_per[0] + split_per[1]]]
    train_l, test_l, valid_l = np.split(label_list, per)
    train_i, test_i, valid_i = np.split(image_list, per)

    if data_split:
        os.makedirs(f'{output_path}/train', exist_ok=True)
        os.makedirs(f'{output_path}/test', exist_ok=True)
        os.makedirs(f'{output_path}/valid', exist_ok=True)
        train = dict(zip(train_l, train_i))
        test = dict(zip(test_l, test_i))
        valid = dict(zip(valid_l, valid_i))
        for tr_k in train.keys():
            with open('{}/train/train_text.txt'.format(output_path), mode='a') as txt:
                txt.writelines('{}\t{}\n'.format(f'{train[tr_k]}.png', tr_k))
        for te_k in test.keys():
            with open('{}/test/test_text.txt'.format(output_path), mode='a') as txt:
                txt.writelines('{}\t{}\n'.format(f'{test[te_k]}.png', te_k))
        for va_k in valid.keys():
            with open('{}/valid/valid_text.txt'.format(output_path), mode='a') as txt:
                txt.writelines('{}\t{}\n'.format(f'{valid[va_k]}.png', va_k))

        print('TRAIN_PER: {:.0%} | TEST_PER: {:.0%} | VALID_PER: {:.0%}'.format(len(train_l) / num, len(test_l) / num,
                                                                                len(valid_l) / num))
        print('TRAIN_NUM: {} | TEST_NUM: {} | VALID_NUM: {}'.format(len(train_i), len(test_i), len(valid_i)))

    else:
        all_dict = dict(zip(label_list, image_list))
        for i, key in enumerate(all_dict.keys()):
            with open('{}/all/all_text.txt'.format(output_path), mode='a') as txt:
                txt.writelines(('{}\t{}\n'.format(f'{all_dict[key]}', key)))
        print('DATA_NUM: {}'.format(len(all_dict)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read SynthText.h5.')
    parser.add_argument('--input-path', required=True, help='Path of SynthText.h5.')
    parser.add_argument('--output-path', required=True, help='Path of output directory.')
    parser.add_argument('--data-split', action='store_true', help='split data')
    parser.add_argument('--split-per', type=str, default='0.6,0.2,0.2', help='split ratio default=0.6,0.2,0.2')
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.split_per, float(args.data_split))
