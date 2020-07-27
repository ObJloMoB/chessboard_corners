import numpy as np
import argparse
import os
import cv2


def main(opts):
    images = np.load(os.path.join(opts.input, f'x{opts.set}.npy'))
    labels = np.load(os.path.join(opts.input, f'y{opts.set}.npy'))
    print(images.shape)
    for i in range(5):
        image = images[i]

        denorm = (labels[i]*image.shape[0]).astype(np.int32)
        print(denorm)

        for j in range(4):
            cv2.circle(image, (denorm[j*2], denorm[j*2+1]), 10, 255, 10)
        cv2.imwrite(f'{i}.jpg', image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input fld',
                        required=True, type=str)
    parser.add_argument('--set', help='train or test',
                        default='train', type=str)

    args = parser.parse_args()
    main(args)