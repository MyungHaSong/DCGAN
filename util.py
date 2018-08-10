import tensorflow as tf
import skimage.io
from glob import glob
from pandas import Series
from os.path import exists
from os import makedirs
from numpy.random import shuffle

def lrelu(x, leak = 0.2):
    return tf.maximum(x, x*0.2)

def dataload(file_list, batch_size):
    shuffle(file_list)
    random = file_list[:batch_size]
    images = skimage.io.imread_collection(random)
    images = images.concatenate()

    try :
        for idx in range(10):
            file_list.pop(idx)

    except IndexError:
        print("next epoch")



    return images, file_list



def makedir(path):
    if not exists(path):
        makedirs(path)
