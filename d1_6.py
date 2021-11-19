#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division, absolute_import

import os
import scipy.io as sio
import tensorflow as tf
import numpy as np
import sys 
import h5py as hp
import math 
import argparse
import matplotlib.pyplot as plt

from functools import partial, update_wrapper
from tensorflow.keras import activations, optimizers, regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Add, Dense, BatchNormalization, Concatenate, Dropout, Subtract, Layer, RepeatVector, Permute, Multiply, LeakyReLU, LayerNormalization
from tensorflow.keras.layers import Flatten, Input, Lambda, Reshape, Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D, UpSampling3D, ConvLSTM2D
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, MaxPool2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, Callback, CSVLogger
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead

from adabelief_tf import AdaBeliefOptimizer
from tlfs import g_loss_npcc
from blocks import SeparableConv3D, ConvEncoderBlock, ConvDecoderBlock
from axial_utils import AxialEncoderBlock, conv1x1


# In[2]:


class ConvEncoder(Layer):
    def __init__(self, enc_filters, separable, reg=1e-4, dropout_rate=0.1):
        super(ConvEncoder, self).__init__()
        
        self.separable = separable
        self.enc_filters = enc_filters
        self.relu = Activation('relu')
        self.dropout = Dropout(dropout_rate)
        
        self.conv0 = SeparableConv3D(conv_filter = self.enc_filters[0], kernel_size = 7, strides = (1,1,2), dilation_rate = (1,1,1),
                                     use_bias = False, separable = self.separable)
        self.bn0 = BatchNormalization()
        self.max0 = MaxPool3D(pool_size = (3,3,3), strides = (1,1,1), padding = 'same')

        self.conv11 = ConvEncoderBlock(filters = [self.enc_filters[0], self.enc_filters[2]], strides = (1,1,1), separable = separable, conv = True)
        self.conv12 = ConvEncoderBlock(filters = [self.enc_filters[0], self.enc_filters[2]], strides = (1,1,1), separable = separable)
        self.conv13 = ConvEncoderBlock(filters = [self.enc_filters[0], self.enc_filters[2]], strides = (1,1,1), separable = separable)
        
        self.conv21 = ConvEncoderBlock(filters = [self.enc_filters[1], self.enc_filters[3]], strides = (2,2,2), separable = separable, conv = True)
        self.conv22 = ConvEncoderBlock(filters = [self.enc_filters[1], self.enc_filters[3]], strides = (1,1,1), separable = separable)
        self.conv23 = ConvEncoderBlock(filters = [self.enc_filters[1], self.enc_filters[3]], strides = (1,1,1), separable = separable)
        self.conv24 = ConvEncoderBlock(filters = [self.enc_filters[1], self.enc_filters[3]], strides = (1,1,1), separable = separable)
        
        self.conv31 = ConvEncoderBlock(filters = [self.enc_filters[2], self.enc_filters[4]], strides = (2,2,2), separable = separable, conv = True)
        self.conv32 = ConvEncoderBlock(filters = [self.enc_filters[2], self.enc_filters[4]], strides = (1,1,1), separable = separable)
        self.conv33 = ConvEncoderBlock(filters = [self.enc_filters[2], self.enc_filters[4]], strides = (1,1,1), separable = separable)
        self.conv34 = ConvEncoderBlock(filters = [self.enc_filters[2], self.enc_filters[4]], strides = (1,1,1), separable = separable)
        self.conv35 = ConvEncoderBlock(filters = [self.enc_filters[2], self.enc_filters[4]], strides = (1,1,1), separable = separable)
        self.conv36 = ConvEncoderBlock(filters = [self.enc_filters[2], self.enc_filters[4]], strides = (1,1,1), separable = separable)
        
    def call(self, x):
        # x: 64 x 64 x 288
        conv0 = self.conv0(x) 
        conv0 = self.bn0(conv0)
        conv0 = self.relu(conv0) # conv0: 64 x 64 x 144
        d0 = self.max0(conv0) # d0: 64 x 64 x 144

        d1 = self.conv11(d0)
        d1 = self.conv12(d1)
        d1 = self.conv13(d1) # d1: 64 x 64 x 144

        d2 = self.conv21(d1)
        d2 = self.conv22(d2)
        d2 = self.conv23(d2)
        d2 = self.conv24(d2) # d2: 32 x 32 x 72

        d3 = self.conv31(d2)
        d3 = self.conv32(d3)
        d3 = self.conv33(d3)
        d3 = self.conv34(d3) 
        d3 = self.conv35(d3)
        d3 = self.conv36(d3) # d3: 16 x 16 x 36
        
#         d4 = self.conv41(d3)
#         d4 = self.conv42(d4)
#         d4 = self.conv43(d4) # d4: 8 x 8 x 18
        
        return d1, d2, d3


# In[3]:


class ConvDecoder(Layer):
    def __init__(self, dec_filters, separable, reg=1e-4, dropout_rate=0.1):
        super(ConvDecoder, self).__init__()
        
        self.dec1 = ConvDecoderBlock(filters = dec_filters[0], separable = separable, upsample_size = (2,2,2), skip_connection = True)
        self.dec2 = ConvDecoderBlock(filters = dec_filters[1], separable = separable, upsample_size = (2,2,2), skip_connection = True)
        self.dec3 = ConvDecoderBlock(filters = dec_filters[2], separable = separable, upsample_size = (2,2,2), skip_connection = False)
        
    def call(self, x, d1, d2):
        x = self.dec1(x, d2)
        x = self.dec2(x, d1)
        x = self.dec3(x, None)
        
        return x


# In[4]:


class AxialTransEncoder(Layer):
    def __init__(self, trans_filters, input_dims):
        super(AxialTransEncoder, self).__init__()
        
        self.downsample1 = Sequential([
            conv1x1(out_planes = trans_filters),
            BatchNormalization(),
        ])
        self.trans1 = AxialEncoderBlock(planes = trans_filters//2, input_dims = input_dims, block_name = 'trans', downsample = self.downsample1)

    def call(self, x):
        x = self.trans1(x)
        
        return x


# In[5]:


class PXCT_Transformer(Layer):
    def __init__(self, separable, output_dims):
        super(PXCT_Transformer, self).__init__()
        self.separable = separable
        self.output_dims = output_dims
        
        self.conv_enc = ConvEncoder(enc_filters = [32, 64, 128, 256, 512], separable = separable)
        self.axial_enc = AxialTransEncoder(trans_filters = 2048, input_dims = [int(x//8) for x in output_dims])
        self.conv_dec = ConvDecoder(dec_filters = [256, 128, 64], separable = separable)
        
    def call(self, x):
        d1, d2, enc = self.conv_enc(x)
        trans = self.axial_enc(enc)
        dec = self.conv_dec(trans, d1, d2)
        
        return dec
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'separable': self.separable,
            'output_dims': self.output_dims
        })
        return config


# In[7]:


lr_callable = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate = 2e-4, decay_steps = 200, power = 0.9)

batch_size = 4
num_epochs = 200

AUTOTUNE = tf.data.AUTOTUNE

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    x = Input((64, 64, 280, 1), batch_size = batch_size)
    out = PXCT_Transformer(separable = False, output_dims = [128, 128, 280])(x)
    model = Model(x, out)
    model.summary(line_length = 90)
    
#     opt_axial_deeplab = Lookahead(RectifiedAdam(warmup_proportion = 0.0))
    opt_panoptic_deeplab = Adam(learning_rate = lr_callable, decay = 0.0)
#     optadabelief = AdaBeliefOptimizer(learning_rate = 5e-4, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-16, 
#                                       weight_decay = 1e-4, rectify = True, amsgrad = False)
#     optadam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-8, decay=1e-4)   
    model.compile(optimizer = opt_panoptic_deeplab, loss=g_loss_npcc, metrics = [g_loss_npcc])


# In[ ]:


ee = sio.loadmat('../source/tr_val_test_indx_list_block_6.mat')
tr_ind = np.squeeze(ee['tr_ind'], axis = 0)
val_ind = np.squeeze(ee['val_ind'], axis = 0)
test_ind = np.squeeze(ee['test_ind'], axis = 0)

with hp.File('../../../../../groups/mit3doptics/fold_slice/training/training_dataset_angle_d1_280px.mat', 'r') as ee:
    training_input = ee['training_input'][()]
    training_output = ee['training_output'][()]

validation_input = np.transpose(training_input[val_ind, :, :, :], (0,3,2,1))
test_input = np.transpose(training_input[test_ind, :, :, :], (0,3,2,1))
training_input = np.transpose(training_input[tr_ind, :, :, :], (0,3,2,1))

validation_output = np.transpose(training_output[val_ind, :, :, :], (0,3,2,1))
test_output = np.transpose(training_output[test_ind, :, :, :], (0,3,2,1))
training_output = np.transpose(training_output[tr_ind, :, :, :], (0,3,2,1))

training_input = np.expand_dims(training_input, axis=-1)
training_output = np.expand_dims(training_output, axis=-1)
validation_input = np.expand_dims(validation_input, axis=-1)
validation_output = np.expand_dims(validation_output, axis=-1)
test_input = np.expand_dims(test_input, axis=-1)
test_output = np.expand_dims(test_output, axis=-1)

print(training_input.shape)
print(training_output.shape)
print(validation_input.shape)
print(validation_output.shape)
print(test_input.shape)
print(test_output.shape)

# shift_and_flip = tf.keras.Sequential([
#     tf.keras.layers.experimental.preprocessing.RandomTranslation((-0.2, 0.2), (-0.2, 0.2), fill_mode = 'constant', fill_value = 0.0)
#     tf.keras.layers.experimental.preprocessing.RandomFlip(mode = "horizontal_and_vertical")     
# ])

def prepare(ds, shuffle = False, augment = False):    
    if shuffle:
        ds = ds.shuffle(1000)
        
    ds = ds.batch(batch_size)
    
#     if augment:
#         ds = ds.map(lambda x, y: (shift_and_flip(x, training = True), y))
        
    return ds.prefetch(buffer_size = AUTOTUNE)

training_dataset = prepare(tf.data.Dataset.from_tensor_slices((training_input, training_output)), shuffle = True, augment = False)
validation_dataset = prepare(tf.data.Dataset.from_tensor_slices((validation_input, validation_output)), shuffle = False, augment = False)
test_dataset = prepare(tf.data.Dataset.from_tensor_slices((test_input, test_output)), shuffle = False, augment = False)
        
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
training_dataset = training_dataset.with_options(options)
validation_dataset = validation_dataset.with_options(options)
test_dataset = test_dataset.with_options(options)


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        loss = self.model.evaluate(self.test_data, verbose=0)
        print('\nTest loss: {}\n'.format(loss))

        
weight_folder = '../weights/d1_6/'
if not os.path.exists(weight_folder):
    os.makedirs(weight_folder)
    
csv_logger = CSVLogger('../log/d1_6.log')
checkpoint = ModelCheckpoint(filepath = weight_folder + '{epoch:03d}.hdf5',
                             monitor='val_loss', verbose=1, save_best_only=False, mode='min')
# reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', min_delta=0.0001,
#                              cooldown=0, min_lr=1e-8)
callbacks_list = [checkpoint, csv_logger, TestCallback(test_dataset)]

# model.load_weights(weight_folder + '004.hdf5')
model.fit(training_dataset, validation_data = validation_dataset, epochs = num_epochs, shuffle = True,
          verbose = 2, callbacks = callbacks_list)

import hdf5storage

# val_rec = model.predict(validation_input, batch_size = batch_size, verbose = 1)
# test_rec = model.predict(test_input, batch_size = batch_size, verbose = 1)
# hdf5storage.writes(mdict = {'test_rec':test_rec, 'test_output':test_output},
#                             filename = '../rec/d1_5_.h5')

