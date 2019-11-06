# Usage: python read_data.py -h
import h5py, os, cv2, numpy as np
def main(input_path, output_path):
    db = h5py.File(input_path)['data']
    os.makedirs(output_path, exist_ok=True)
    for k in db.keys():
        d = db[k]
        print(f'extracting {k}...')
        im = d[:]
        cv2.imwrite(f'{output_path}/{k}.png', im)
        if 'count' in d.attrs:
            c = d.attrs['count']
            for i in range(c):
                print(f'instance {i}')
                print('character bounding boxes (2 x 4 x N):')
                bbs = d.attrs[f'charBB{i}']
                print(bbs)
                print('text:')
                print(''.join([x.decode('utf8') for x in d.attrs[f'txt{i}']]))
                cv2.polylines(im, np.around(bbs.transpose((2, 1, 0))).astype(np.int), True, (255, 255, 255))
        else:
            print('character bounding boxes (2 x 4 x N):')
            bbs = d.attrs['charBB']
            print(bbs)
            print('text:')
            print(''.join([x.decode('utf8') for x in d.attrs['txt']]))
            cv2.polylines(im, np.around(bbs.transpose((2, 1, 0))).astype(np.int), True, (255, 255, 255))
        cv2.imwrite(f'{output_path}/{k}_annotation.png', im)
        print('label:')
        print(['train', 'val', 'test'][d.attrs['label']])
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read SynthText.h5.')
    parser.add_argument('--input-path', required=True, help='Path of SynthText.h5, the result of main.py.')
    parser.add_argument('--output-path', required=True, help='Path of output directory.')
    args = parser.parse_args()
    main(args.input_path, args.output_path)
