import numpy as np
import argparse
import torch
from lib.model import Model
from lib.dataset import get_test
import os


def main(opts):
    # Select device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Define model
    model = Model().to(device)
    model.load_state_dict(torch.load(opts.weights))
    model.eval()

    # Define dataloader
    test_loader = get_test(opts.input)

    preds = []
    for img in test_loader:
        img = img.to(device)
        with torch.no_grad():
            predict = model(img)
            predict = predict.cpu().detach().numpy().squeeze()
        preds.append(predict)
    preds = np.array(preds)
    np.save(os.path.join(opts.input, 'ytest.npy'), preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input fld',
                        required=True, type=str)
    parser.add_argument('--weights', help='DATA',
                        default='weights/checkpoint.pth', type=str)

    args = parser.parse_args()
    main(args)