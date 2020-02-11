import os
import gzip
import numpy as np
import time

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16)
    print(images.shape)
    images = images.reshape(len(labels), 28, 28)

    # for flat arrays:
    # with gzip.open(images_path, 'rb') as imgpath:
    #     images = np.frombuffer(imgpath.read(), dtype=np.uint8,
    #                            offset=16).reshape(len(labels), 784)

    return images, labels
