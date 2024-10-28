"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(filename="./data/training.log", filemode="w", level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx
import numpy as np
import gzip, struct

'''
def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(download_file(base_url+label, os.path.join('data',label))) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_file(base_url+image, os.path.join('data',image)), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)
'''

if __name__ == '__main__':

    data_dir = "/data/"
    train_fname = data_dir + 'mnist_train.rec'
    #val_fname = data_dir + 'cifar10_val.rec'
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--num-classes', type=int, default=10,
    #                    help='the number of classes')
    #parser.add_argument('--num-examples', type=int, default=60000,
    #                    help='the number of training examples')
'''
usage: train_mnist.py [-h] [--num-classes NUM_CLASSES]
                      [--num-examples NUM_EXAMPLES] [--network NETWORK]
                      [--num-layers NUM_LAYERS] [--gpus GPUS]
                      [--kv-store KV_STORE] [--num-epochs NUM_EPOCHS]
                      [--lr LR] [--lr-factor LR_FACTOR]
                      [--lr-step-epochs LR_STEP_EPOCHS]
                      [--optimizer OPTIMIZER] [--mom MOM] [--wd WD]
                      [--batch-size BATCH_SIZE] [--disp-batches DISP_BATCHES]
                     # [--model-prefix MODEL_PREFIX] [--monitor MONITOR]
                      [--load-epoch LOAD_EPOCH] [--top-k TOP_K]
                      [--test-io TEST_IO] [--dtype DTYPE]
train_mnist.py: error: unrecognized arguments: --data-train /data/mnist_train.rec

'''

    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 3)

    num_examples = 60000
    batch_size = 128
    disp_batches = 10
    
    parser.set_defaults(
        # network
        network        = 'resnet',
        num_layers     = 110,
        # data
        data_train     = train_fname,
        #data_val       = val_fname,
        num_classes    = 10,
        num_examples  = num_examples,
        image_shape    = '1,28,28',
        # train
        gpus           = None,
        batch_size     = batch_size,
        disp_batches   = disp_batches,
        num_epochs     = 20,
        lr             = .05,
        lr_step_epochs = '10',
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)
