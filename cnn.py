import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main
from PIL import Image


# Form randomly permuted batches of training data
def get_mini_batch(im_train, label_train, batch_size):
    # Reshape images so each row contains pixel data
    im_train = im_train.transpose() # 12000 x 196 (14x14)
    label_train = label_train.transpose() # 12000 x 1 (written number)

    # shuffle the im_train and label_train arrays
    num_images, num_pixels = im_train.shape
    rand_indices = np.random.permutation(num_images)
    rand_im_train = im_train[rand_indices]
    rand_label_train = label_train[rand_indices]

    # partition arrays into batches
    mini_batch_x = []
    mini_batch_y = []
    num_batches = math.floor(num_images / batch_size)
    for i in range(num_batches):
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            image = rand_im_train[(i*batch_size) + j]
            batch_x.append(image)
            # construct one-hot encoding from training label
            onehot_label = [0]*10
            onehot_label[rand_label_train[(i*batch_size) + j][0]] = 1
            batch_y.append(onehot_label)
        # Add current batch to the set of batches
        mini_batch_x.append(batch_x)
        mini_batch_y.append(batch_y)

    # get remaining images
    batch_x = []
    batch_y = []
    place = (num_batches-1)*batch_size
    for i in range(place, num_images):
        image = rand_im_train[i]
        batch_x.append(image)
        onehot_label = [0]*10
        onehot_label[rand_label_train[i][0]] = 1
        batch_y.append(onehot_label)

    mini_batch_x.append(batch_x)
    mini_batch_y.append(batch_y)

    # reformat array for return shape (each col is an image/label)
    # output img shape: (196 x batch_size)
    # output label shape: (10 x batch_size) -- one hot enc
    final_mini_batch_x = []
    final_mini_batch_y = []
    for bx, by in zip(mini_batch_x,mini_batch_y):
        x = np.asarray(bx).transpose()
        y = np.asarray(by).transpose()
        final_mini_batch_x.append(x.tolist())
        final_mini_batch_y.append(y.tolist())

    mini_batch_x = final_mini_batch_x
    mini_batch_y = final_mini_batch_y

    return mini_batch_x, mini_batch_y


# x: input to fully connected layer (m x 1)
# w: weights (n x m)
# b: bias (n x 1)
# Output: y (n x 1)
def fc(x, w, b):
    # (n x m) * (m x 1) = (n x 1)
    y = (np.dot(w,x)) + b
    return y


# Input: dl_dy (1 x n) -- loss deriviative w.r.t model prediction y
def fc_backward(dl_dy, x, w):
    # loss derivative with respect to the input x
    # (1 x n) * (n x m) = (1 x m)
    # y = (x * w) + b
    # dl/dx = dl/dy * dy/dx
    # dl/dx = dl/dy * w
    dl_dx = np.dot(dl_dy, w)
    # loss derivative with respect to the weights
    # dl/dw = dl/dy * dy/dw
    # dl/dw = dl/dy * x

    dl_dw = np.dot(np.array([dl_dy]).transpose(), x.transpose()).flatten()

    # loss derivative with respect to the bias
    # dl/db = dl/dy * dy/db
    dl_db = dl_dy

    #Shape:(1xm) (1x(nxm)) (1xn)
    return dl_dx, dl_dw, dl_db


# Inputs: y_tilde (prediction), y (ground truth)
# Outputs: l (L2 loss), dl_dy (loss derivative w.r.t. prediction)
def loss_euclidean(y_tilde, y):
    # compute L2 loss (euclidean distance)
    l = (np.linalg.norm(y - y_tilde))**2
    # compute derivative of L2 loss function w.r.t. the prediction y_tilde
    dl_dy = (2*(y_tilde - y)).flatten(order = 'F')
    return l, dl_dy


# run output through softmax and compute cross entropy loss
def loss_cross_entropy_softmax(y_tilde, y):
    # compute softmax output of model
    softmax = np.exp(y_tilde) / np.sum(np.exp(y_tilde))
    # compute cross entropy loss with ground truth
    l = np.sum(y * np.log(softmax))
    # compute derivative of cross entropy loss with shorthand trick: https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    dl_dy = (softmax - y).flatten(order = 'F')
    return l, dl_dy


# Leaky ReLu implementation
def relu(x):
    e = 0.01
    y = np.where(x > 0, x, x * e)
    return y


# input: loss derivative w.r.t output (dl_dy)
# output: loss derivative w.r.t. input (x)
def relu_backward(dl_dy, x):
    e = 0.01
    # compute derivative of leaky relu w.r.t. x
    dy_dx = np.where(x >= 0, 1, e)

    # compute derivative of loss w.r.t. x
    dl_dx = dl_dy * dy_dx
    return dl_dx


'''
--------
Code for im2col provided by: https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
--------
'''
def im2col(x,hh,ww,stride):
    c,h,w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col


# input: 14x14 image, weights, biases
def conv(x, w_conv, b_conv):
    rows, cols, channels = x.shape

    # pad x with a border of zeros
    x = np.pad(x, 1).transpose()[1].transpose()
    x = np.asarray([x])

    # x_col is the transformation of input to region columns
    # 1) flatten each window into columns of matrix
    # 2) flatten kernel into row vector
    # 3) matrix multiply
    x_col = im2col(x, 3, 3, 1)

    # flatten kernel, multiply it with the columnized input to get conv output
    convolved_x = np.dot(x_col, np.reshape(w_conv, (9,3), order = 'F'))
    
    # reshape into feature map image output
    y = np.reshape(convolved_x, (rows, cols, w_conv.shape[3]), order = 'F')
 
    # include the bias
    for i in range(rows):
        for j in range(cols):
            for k in range(w_conv.shape[3]):
                y[i][j][k] += b_conv[k]

    return y


# ONE BIAS PER FEATURE MAP (3, in this case)
# input: loss derivative w.r.t output (dl_dy)
# output: loss derivative w.r.t conv weights and biases
def conv_backward(dl_dy, x, w_conv, b_conv):
    # initialize the loss derivatives w.r.t. the convolutional weights and biases
    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)

    # pad the input
    pad = np.pad(x, 1).transpose()[1].transpose()
    pad = np.asarray([pad])

    # perform convolution with existing im2col() function
    # x_col is the transformation of input to region columns
    # 1) flatten each window into columns of matrix
    # 2) flatten kernel into row vector
    # 3) matrix multiply
    x_col = im2col(pad, 3, 3, 1)
    dl_dw = np.reshape(np.dot(x_col.transpose(), np.reshape(dl_dy, (196,3), order = 'F')), w_conv.shape, order = 'F')
    # compute loss gradient with respect to the bias
    # dl/db = dl/dy * dy/db
    print(dl_db)
    exit(0)
    for i in range(dl_db.shape[0]):
        dl_db[i][0] = np.sum(dl_dy[i])

    return dl_dw, dl_db


# reduces size of feature maps: feature abstraction + denoising + easier computation
def pool2x2(x):
    rows, cols, channels = x.shape
    y = np.zeros((rows//2, cols//2, channels))
    # iterate through dimensions of x and fill y with max values for each 2x2 block
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            # r/c indices into pooled output image
            y_row = (i + 1)//2
            y_col = (j + 1)//2
            for q in range(channels):
                # find max of 2x2 square within channel q
                y_val = max(x[i][j][q], x[i+1][j][q], x[i][j+1][q], x[i+1][j+1][q])
                y[y_row][y_col][q] = y_val
    return y


# input: loss derivative w.r.t. output (dl_dy)
# output: loss derivative w.r.t. input (x)
def pool2x2_backward(dl_dy, x):
    rows, cols, channels = x.shape
    dl_dx = np.zeros(x.shape)
    # dl_dx will contain all zeros with the exception of the dl_dy values in...
    # .. the position where x contains the max value in each 2x2 block
    for row in range(0, rows, 2):
        for col in range(0, cols, 2):
            y_row = (row + 1)//2
            y_col = (col + 1)//2
            for q in range(channels):
                max_val = x[y_row][y_col][q]
                dl_dy_val = dl_dy[y_row][y_col][q]

                # iterate through each position in the 2x2 block
                for i in range(2):
                    for j in range(2):
                        if x[i][j][q] == max_val:
                            dl_dx[i][j][q] = dl_dy_val
    return dl_dx


def flattening(x):
    flat = x.flatten(order = 'F')
    y = np.asarray([flat]).transpose()
    return y


def flattening_backward(dl_dy, x, y):
    dl_dx = np.reshape(dl_dy, x.shape, order = 'F')
    return dl_dx


def train_slp(mini_batch_x, mini_batch_y, mode = 'softmax'):
    batch_size = len(mini_batch_x[0][0])

    # set learning rate and decay rate
    lr = 0.2
    dr = 0.3

    # initialize weights and biases (random)
    pixel_weights = np.random.rand(10, 196)
    bias_weights = np.random.rand(10, 1)

    k = 0 # current batch index
    # iterate over 5000 batches
    for i in range(5000):
        # adjust the learning rate with decay rate
        if (i % 1000) == 0:
            lr *= dr

        # Initialize weight and bias loss derivatives
        dL_dw = dL_db = 0

        batch_x = np.asarray(mini_batch_x[k])
        batch_y = np.asarray(mini_batch_y[k])

        # Iterate through each image in the k'th mini-batch
        for j in range(batch_size):
            # get image and label from batch (j'th column)
            image = np.asarray([batch_x[:, j]]).transpose()
            true_label = np.asarray([batch_y[:, j]]).transpose()

            # predict image label
            predicted_label = fc(image, pixel_weights, bias_weights)

            # compute loss
            if mode == 'softmax':
                l, dl_dy = loss_cross_entropy_softmax(predicted_label, true_label)
            else:
                l, dl_dy = loss_euclidean(predicted_label, true_label)

            # gradient back-propagation; compute loss gradients
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, image, pixel_weights)

            # accumulate weight and bias adjustments for each training example in batch
            dL_dw += dl_dw
            dL_db += dl_db

        # move to the next mini-batch, or cycle back to first mini-batch if reached end
        k += 1
        if k >= len(mini_batch_x):
            k = 0

        # update weights and biases for batch from accumulated adjustments
        rs_dL_dw = np.reshape(dL_dw, pixel_weights.shape)
        gamma_r = (lr / batch_size)
        pixel_weights -= (gamma_r * rs_dL_dw)
        bias_weights -= (gamma_r * np.array([dL_db]).transpose())

    return pixel_weights, bias_weights



def train_mlp(mini_batch_x, mini_batch_y):
    batch_size = len(mini_batch_x[0][0])

    # set learning rate and decay rate
    lr = 0.05
    dr = 0.9

    # initialize weights and biases
    pixel_weights1 = np.random.rand(30, 196)
    bias_weights1 = np.random.rand(30, 1)
    pixel_weights2 = np.random.rand(10, 30)
    bias_weights2 = np.random.rand(10, 1)

    k = 0
    for i in range(10000):
        # adjust the learning rate with decay rate
        if (i % 1000) == 0:
            lr *= dr

        # Initialize weight and bias loss derivatives
        dL_dw_layer1 = dL_db_layer1 = dL_dw_layer2 = dL_db_layer2 = 0

        batch_x = np.asarray(mini_batch_x[k])
        batch_y = np.asarray(mini_batch_y[k])

        # Iterate through each image in the k'th mini-batch
        for j in range(batch_size):
            # get image and label from batch (j'th column)
            image = np.asarray([batch_x[:, j]]).transpose()
            true_label = np.asarray([batch_y[:, j]]).transpose()

            # predict image label
            output_layer1 = fc(image, pixel_weights1, bias_weights1)
            relu_layer1 = relu(output_layer1)
            output_layer2 = fc(relu_layer1, pixel_weights2, bias_weights2)

            # compute loss
            l, dl_dy = loss_cross_entropy_softmax(output_layer2, true_label)

            # dy/dx2 = d(xw + b) / dx
            # dl_dx2 = dl/dy * w

            # gradient back-propagation; compute loss gradients
            dl_dx_layer2, dl_dw_layer2, dl_db_layer2 = fc_backward(dl_dy, relu_layer1, pixel_weights2)
            dl_dx = relu_backward(np.array([dl_dx_layer2]).transpose(), relu_layer1)
            dl_dx_layer1, dl_dw_layer1, dl_db_layer1 = fc_backward(dl_dx.flatten(), image, pixel_weights1)

            # accumulate loss gradients
            dL_dw_layer1 += dl_dw_layer1
            dL_db_layer1 += dl_db_layer1
            dL_dw_layer2 += dl_dw_layer2
            dL_db_layer2 += dl_db_layer2

        # move to the next mini-batch, or cycle back to first mini-batch if reached end
        k += 1
        if k >= len(mini_batch_x):
            k = 0

        # update weights and biases
        rs_dL_dw_layer1 = np.reshape(dL_dw_layer1, pixel_weights1.shape)
        rs_dL_dw_layer2 = np.reshape(dL_dw_layer2, pixel_weights2.shape)
        gamma_r = (lr / batch_size)
        pixel_weights1 -= (gamma_r * rs_dL_dw_layer1)
        bias_weights1 -= (gamma_r * np.array([dL_db_layer1]).transpose())
        pixel_weights2 -= (gamma_r * rs_dL_dw_layer2)
        bias_weights2 -= (gamma_r * np.array([dL_db_layer2]).transpose())

    return pixel_weights1, bias_weights1, pixel_weights2, bias_weights2



def train_cnn(mini_batch_x, mini_batch_y):
    # set learning rate and decay rate
    lr = 0.1
    dr = 0.85

    # initialize weights and biases
    conv_weights = np.random.rand(3, 3, 1, 3)
    conv_biases = np.random.rand(3, 1)
    fc_weights = np.random.rand(10, 147)
    fc_biases = np.random.rand(10, 1)

    batch_size = len(mini_batch_x[0][0])
    k = 0
    for i in range(10000):
        # adjust the learning rate with decay rate
        if (i % 1000) == 0:
            print(i)
            lr *= dr

        # Initialize weight and bias loss derivatives
        dL_dw_conv_layer = dL_db_conv_layer = dL_dw_fc_layer = dL_db_fc_layer = 0

        batch_x = np.asarray(mini_batch_x[k])
        batch_y = np.asarray(mini_batch_y[k])

        # Iterate through each image in the k'th mini-batch
        for j in range(batch_size):
            # get image and label from batch (j'th column)
            image = np.asarray([batch_x[:, j]]).transpose()
            true_label = np.asarray([batch_y[:, j]]).transpose()

            # Input layer:
            image = np.asarray([[row[j]] for row in mini_batch_x[k]])
            reshaped_image = np.reshape(image, (14,14,1), order = 'F')
            # Convolutional 3x3 layer
            conv_x = conv(reshaped_image, conv_weights, conv_biases)
            # ReLu layer
            f = relu(conv_x)
            # Max-pooling 2x2 layer
            pool_f = pool2x2(f)
            # Vectorize image as input for FC layer
            flat = flattening(pool_f)
            # Forward pass vector through fully-connected layer (10)
            fc_layer = fc(flat, fc_weights, fc_biases)
            #Soft-max
            l, dl_dy_fc = loss_cross_entropy_softmax(fc_layer, true_label)
            # gradient back-propagation; compute loss gradients
            dl_df_flat, dl_dw_fc, dl_db_fc = fc_backward(dl_dy_fc, flat, fc_weights)
            # reshape gradients back to a square matrix to continue backprop
            dl_d_pool = flattening_backward(dl_df_flat, pool_f, flat)
            # unpool with zeros around original max vals
            dl_df = pool2x2_backward(dl_d_pool, f)
            dl_d_conv = relu_backward(dl_df, f)

            dl_dw_conv, dl_db_conv = conv_backward(dl_d_conv, reshaped_image, conv_weights, conv_biases)

            dL_dw_conv_layer += dl_dw_conv
            dL_db_conv_layer += dl_db_conv
            dL_dw_fc_layer += dl_dw_fc
            dL_db_fc_layer += dl_db_fc

        # move to the next mini-batch, or cycle back to first mini-batch if reached end
        k += 1
        if k >= len(mini_batch_x):
            k = 0

        # update weights and biases
        reshaped_dL_dw_fc_layer = np.reshape(dL_dw_fc_layer, fc_weights.shape)
        gamma_r = (lr/batch_size)
        conv_weights -= (gamma_r * dL_dw_conv_layer)
        conv_biases -= (gamma_r * dL_db_conv_layer)
        fc_weights -= (gamma_r * reshaped_dL_dw_fc_layer)
        fc_biases -= np.array([gamma_r * dL_db_fc_layer]).transpose()

    w_conv = conv_weights
    b_conv = conv_biases
    w_fc = fc_weights
    b_fc = fc_biases

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    # main.main_slp_linear()
    # main.main_slp()
    # main.main_mlp()
    main.main_cnn()
