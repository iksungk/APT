#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras import backend as K

# import hdf5storage
import os
import scipy.io as sio
import tensorflow as tf
import numpy as np

def normalize(y_inp, const):
    mu_inp = K.mean(y_inp, axis = (1,2,3,4), keepdims = True)
    std_inp = K.std(y_inp, axis = (1,2,3,4), keepdims = True)

    output = (y_inp - mu_inp) / (const * std_inp) + 0.5

    return output
    
    
def ssim_loss(y_pred, y_true):
    # y_pred: B x Ny x Nx x Nz x 1
    # y_true: B x Ny x Nx x Nz x 1
    const = 10
    y_pred = normalize(y_pred, const)
    y_true = normalize(y_true, const)
    
    u_pred = K.mean(y_pred, axis = (1,2,3,4), keepdims = True)
    u_true = K.mean(y_true, axis = (1,2,3,4), keepdims = True)
    fsp = y_pred - u_pred
    fst = y_true - u_true
    
    var_pred = K.var(y_pred, axis = (1,2,3,4))
    var_true = K.var(y_true, axis = (1,2,3,4))
    
    L = 1
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    
    ret = tf.squeeze(2 * u_true * u_pred + c1, axis = (1,2,3,4)) / tf.squeeze(u_true ** 2 + u_pred ** 2 + c1, axis = (1,2,3,4))
    cs = (2 * K.mean(fsp * fst, axis=(1,2,3,4)) + c2) / (var_pred + var_true + c2)

    return ret, cs
    # return (-1) * ret * cs

    
def multissim_loss(y_pred, y_true):
    # y_pred: B x Ny x Nx x Nz x 1
    # y_true: B x Ny x Nx x Nz x 1
    weights = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype = tf.float32)
    ssims = []
    mcs = []
    levels = weights.shape[0]
    
    sim, cs = ssim_loss(y_pred, y_true)
    ssims.append(sim)
    mcs.append(cs)
    
    for _ in range(levels - 1):
        y_pred = K.pool3d(y_pred, pool_size = (2,2,2), strides=(2,2,2), padding='same', pool_mode='avg')
        y_true = K.pool3d(y_true, pool_size = (2,2,2), strides=(2,2,2), padding='same', pool_mode='avg')
        sim, cs = ssim_loss(y_pred, y_true)
        ssims.append(sim)
        mcs.append(cs)

    ssims = K.permute_dimensions(K.stack(ssims), (1,0))
    mcs = K.permute_dimensions(K.stack(mcs), (1,0))
    
    pow1 = mcs ** K.repeat_elements(K.expand_dims(weights, axis = 0), axis = 0, rep = ssims.shape[0])
    pow2 = ssims ** K.repeat_elements(K.expand_dims(weights, axis = 0), axis = 0, rep = ssims.shape[0])
    
    output = K.prod(pow1, axis = -1) * pow2[:,-1]
    
    return (-1) * output


def total_npcc_loss(y_pred, y_true):
    npcc_loss_h = height_npcc_loss(y_pred, y_true)
    npcc_loss_w = width_npcc_loss(y_pred, y_true)
    npcc_loss_l = layer_npcc_loss(y_pred, y_true)
    
    npcc_loss = 1/3 * (npcc_loss_h + npcc_loss_w + npcc_loss_l)
    
    return npcc_loss


def height_npcc_loss(y_pred, y_true):
    nom_pred = y_pred - K.mean(y_pred, axis = (2,3,4), keepdims = True) # B x Ny x Nx x Nz x 1 -> B x Ny x 1 x 1 x 1
    nom_true = y_true - K.mean(y_true, axis = (2,3,4), keepdims = True) # B x Ny x Nx x Nz x 1 -> B x Ny x 1 x 1 x 1
    nom = K.mean(nom_pred * nom_true, axis = (2,3,4)) # B x Nz

    den_pred = K.std(y_pred, axis = (2,3,4)) # B x Ny x Nx x Nz x 1 -> B x Nz
    den_true = K.std(y_true, axis = (2,3,4)) # B x Ny x Nx x Nz x 1 -> B x Nz
#     den = K.clip(den_pred * den_true, K.epsilon(), None) # B x Nz
    den = den_pred * den_true
    
    npcc_loss = (-1) * K.mean(nom / den, axis = 1) # B

    return npcc_loss

    
def width_npcc_loss(y_pred, y_true):
    nom_pred = y_pred - K.mean(y_pred, axis = (1,3,4), keepdims = True) # B x Ny x Nx x Nz x 1 -> B x 1 x Nx x 1 x 1
    nom_true = y_true - K.mean(y_true, axis = (1,3,4), keepdims = True) # B x Ny x Nx x Nz x 1 -> B x 1 x Nx x 1 x 1
    nom = K.mean(nom_pred * nom_true, axis = (1,3,4)) # B x Nz

    den_pred = K.std(y_pred, axis = (1,3,4)) # B x Ny x Nx x Nz x 1 -> B x Nz
    den_true = K.std(y_true, axis = (1,3,4)) # B x Ny x Nx x Nz x 1 -> B x Nz
#     den = K.clip(den_pred * den_true, K.epsilon(), None) # B x Nz
    den = den_pred * den_true
    
    npcc_loss = (-1) * K.mean(nom / den, axis = 1) # B

    return npcc_loss


def layer_npcc_loss(y_pred, y_true):
    nom_pred = y_pred - K.mean(y_pred, axis = (1,2,4), keepdims = True) # B x Ny x Nx x Nz x 1 -> B x 1 x 1 x Nz x 1
    nom_true = y_true - K.mean(y_true, axis = (1,2,4), keepdims = True) # B x Ny x Nx x Nz x 1 -> B x 1 x 1 x Nz x 1
    nom = K.mean(nom_pred * nom_true, axis = (1,2,4)) # B x Nz

    den_pred = K.std(y_pred, axis = (1,2,4)) # B x Ny x Nx x Nz x 1 -> B x Nz
    den_true = K.std(y_true, axis = (1,2,4)) # B x Ny x Nx x Nz x 1 -> B x Nz
#     den = K.clip(den_pred * den_true, K.epsilon(), None) # B x Nz
    den = den_pred * den_true

    weights = tf.concat([tf.constant(10.0, shape = (nom.shape[0], 112)), tf.constant(1.0, shape = (nom.shape[0], 176))], axis = 1)
    weights /= tf.math.reduce_sum(weights)
    
    npcc_loss = (-1) * K.sum(weights * nom / den, axis = 1) # B

    return npcc_loss


def split_g_loss_npcc(y_pred, y_true):
    y_pred_f = y_pred[:, :, :, 0:112, :]
    y_pred_c = y_pred[:, :, :, 112::, :]
    y_true_f = y_true[:, :, :, 0:112, :]
    y_true_c = y_true[:, :, :, 112::, :]
    
    npcc_loss_f = g_loss_npcc(y_pred_f, y_true_f)
    npcc_loss_c = g_loss_npcc(y_pred_c, y_true_c)

    npcc_loss = 0.9 * npcc_loss_f + 0.1 * npcc_loss_c
    
    return npcc_loss

    
def g_loss_npcc(y_pred, y_true):
    fsp = y_pred - K.mean(y_pred, axis=(1,2,3,4), keepdims=True)
    fst = y_true - K.mean(y_true, axis=(1,2,3,4), keepdims=True)

    devP = K.std(y_pred, axis = (1,2,3,4))
    devT = K.std(y_true, axis = (1,2,3,4))
    
    npcc_loss = (-1) * K.mean(fsp * fst, axis = (1,2,3,4)) / K.clip(devP * devT, K.epsilon(), None)    ## (BL,1)
    return npcc_loss


# In[ ]:




