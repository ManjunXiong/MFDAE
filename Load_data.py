import numpy as np
from keras_preprocessing import image
from PIL import Image
from numpy import hstack
from scipy import misc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.preprocessing import normalize
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from MFDAE import gaussian_noise_layer

import warnings
warnings.filterwarnings("ignore")

path = './data'


def CWRU_V2():
    data = scio.loadmat("C:\\Users\\18227\\Desktop\\multi_view\\DEMVC\\DEMVC2\\DEMVC-master\\data\\CWRU_V2.mat")
    print("dataset CWRU_v2")
    x1 = data['X2']
    x2 = data['X1']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    Y = Y.reshape(15000, )
    # print(Y1)
    # print(Y1.shape)
    Y = Y[index]
    Y = Y.reshape(1, 15000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2], Y
    # return [x1], Y

def CWRU_V3():
    data = scio.loadmat("C:\\Users\\18227\\Desktop\\multi_view\\DEMVC\\DEMVC2\\DEMVC-master\\data\\CWRU_V3.mat")
    print("dataset CWRU_V3")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x3 = x3[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    x3 = gaussian_noise_layer(x3, 0.001)
    Y = Y.reshape(15000, )
    Y = Y[index]
    Y = Y.reshape(1, 15000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2,x3], Y

def N_CWRU_V3():
    data = scio.loadmat("C:\\Users\\18227\\Desktop\\multi_view\\DEMVC\\DEMVC2\\DEMVC-master\\data\\N_CWRU_V3.mat")
    print("dataset N_CWRU_V3")
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x3 = x3[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    x3 = gaussian_noise_layer(x3, 0.001)
    Y = Y.reshape(6000, )
    Y = Y[index]
    Y = Y.reshape(1, 6000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2,x3], Y

def N_CWRU_V2():
    data = scio.loadmat("C:\\Users\\18227\\Desktop\\multi_view\\DEMVC\\DEMVC2\\DEMVC-master\\data\\N_CWRU_V2.mat")
    print("dataset N_CWRU_V3")
    x1 = data['X1']
    x2 = data['X2']
    Y = data['Y'][0]
    index = [i for i in range(len(x1))]
    np.random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    x1 = gaussian_noise_layer(x1, 0.001)
    x2 = gaussian_noise_layer(x2, 0.001)
    Y = Y.reshape(6000, )
    Y = Y[index]
    Y = Y.reshape(1, 6000)
    Y = Y[0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2], Y


def load_data_conv(dataset):
    print("load:", dataset)#load: MNIST_USPS_COMIC
    if dataset == 'CWRU_V3':
        return CWRU_V3()
    elif dataset == 'N_CWRU_V3':
        return N_CWRU_V3()
    elif dataset == 'N_CWRU_V2':
        return N_CWRU_V2()
    elif dataset == 'CWRU_V2':
        return CWRU_V2()
    else:
        raise ValueError('Not defined for loading %s' % dataset)
