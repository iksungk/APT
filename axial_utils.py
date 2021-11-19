import tensorflow as tf
from tensorflow.keras import activations, optimizers, regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Add, Dense, BatchNormalization, Concatenate, Dropout, Subtract, Layer, RepeatVector, Permute, Multiply, LayerNormalization
from tensorflow.keras.layers import Flatten, Input, Lambda, Reshape, Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D, UpSampling3D
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, MaxPool2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.initializers import RandomNormal


def conv1x1(out_planes, strides = 1):
    return Conv3D(filters = out_planes, kernel_size = 1, strides = strides, use_bias = False)


class AxialAttention(Layer):
    def __init__(self, att_axis, groups, in_planes, out_planes, kernel_size, att_name, strides = 1):
        super(AxialAttention, self).__init__()
        
        self.att_axis = att_axis
        self.groups = groups
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.strides = strides
        
        self.bn_qkv = BatchNormalization()
        self.bn_similarity = BatchNormalization()
        self.bn_output = BatchNormalization()
        
        self.qkv_transform = Conv1D(filters = 2 * out_planes, kernel_size = 1, strides = 1, padding = 'same', use_bias = False, data_format = "channels_first",
                                    kernel_initializer = tf.keras.initializers.RandomNormal(0.0, tf.math.sqrt(1.0 / self.in_planes)))
        
        self.relative = tf.Variable(tf.random.normal(shape = (self.group_planes * 2, kernel_size * 2 - 1), mean = 0.0, stddev = tf.math.sqrt(1.0 / self.group_planes)), 
                                    trainable = True, name = 'rel_' + att_name)
        query_index = tf.expand_dims(tf.range(kernel_size), axis = 0)
        key_index = tf.expand_dims(tf.range(kernel_size), axis = 1)
        relative_index = key_index - query_index + kernel_size - 1
        self.flatten_index = tf.reshape(relative_index, -1)
        
        if self.strides > 1:
            self.pooling = AveragePooling3D(pool_size = (strides, strides, strides), strides = strides)
        
    def call(self, x):
        # x: B x H x W x L x C
        if self.att_axis == 'H':
            x = tf.transpose(x, [0, 2, 3, 4, 1])
            
        elif self.att_axis == 'W':
            x = tf.transpose(x, [0, 1, 3, 4, 2])
            
        elif self.att_axis == 'L':
            x = tf.transpose(x, [0, 1, 2, 4, 3])
            
        # Let self.att_axis == 'H', then x: B x W x L x C x H
        B, W, L, C, H = x.shape
        x = tf.reshape(x, [B * W * L, C, H])
        
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = tf.split(tf.reshape(qkv, [B * W * L, self.groups, self.group_planes * 2, H]), 
                           [self.group_planes // 2, self.group_planes // 2, self.group_planes], axis = -2)
    
        all_embeddings = tf.reshape(tf.gather(self.relative, self.flatten_index, axis = 1), [self.group_planes * 2, self.kernel_size, self.kernel_size])
        q_embedding, k_embedding, v_embedding = tf.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], axis = 0)
        
        # q: BWL x Nh x dqh x H, q_embedding: dqh x H x H -> qr: BWL x Nh x H x H.
        qr = tf.einsum('bgci,cij -> bgij', q, q_embedding)
        kr = tf.transpose(tf.einsum('bgci,cij -> bgij', k, k_embedding), [0, 1, 3, 2])
        qk = tf.einsum('bgci, bgcj -> bgij', qr, kr)
        
        stacked_similarity = tf.concat([qk, qr, kr], axis = 1)
        stacked_similarity = tf.math.reduce_sum(tf.reshape(self.bn_similarity(stacked_similarity), [B * W * L, 3, self.groups, H, H]), axis = 1)
        
        similarity = tf.nn.softmax(stacked_similarity, axis = 3)
        sv = tf.einsum('bgij,bgcj -> bgci', similarity, v)
        sve = tf.einsum('bgij,cij -> bgci', similarity, v_embedding)
        stacked_output = tf.reshape(tf.concat([sv, sve], axis = -1), [B * W * L, self.out_planes * 2, H])
        output = tf.math.reduce_sum(tf.reshape(self.bn_output(stacked_output), [B, W, L, self.out_planes, 2, H]), axis = -2)
        
        if self.att_axis == 'H':
            output = tf.transpose(output, [0, 4, 1, 2, 3])
            
        elif self.att_axis == 'W':
            output = tf.transpose(output, [0, 1, 4, 2, 3])
            
        elif self.att_axis == 'L':
            output = tf.transpose(output, [0, 1, 2, 4, 3])
        
        if self.strides > 1:
            output = self.pooling(output)
            
        return output
    
    
class AxialEncoderBlock(Layer):
    expansion = 2
    
    def __init__(self, planes, input_dims, block_name, strides = 1, groups = 8, dilation = 1, downsample = None, base_width = 64, return_skip = False):
        super(AxialEncoderBlock, self).__init__()
                
        filters = int(planes * (base_width / 64.))
        self.conv_down = conv1x1(filters)
        self.bn1 = BatchNormalization()
        self.height_block = AxialAttention(att_axis = 'H', groups = groups, in_planes = filters, out_planes = filters, kernel_size = input_dims[0], att_name = block_name + '_h')
        self.width_block = AxialAttention(att_axis = 'W', groups = groups, in_planes = filters, out_planes = filters, kernel_size = input_dims[1], att_name = block_name + '_w')
        self.layer_block = AxialAttention(att_axis = 'L', groups = groups, in_planes = filters, out_planes = filters, kernel_size = input_dims[2], att_name = block_name + '_l',
                                          strides = strides)
        self.conv_up = conv1x1(planes * self.expansion)
        self.bn2 = BatchNormalization()
        self.relu = Activation('relu')
        self.downsample = downsample
        self.return_skip = return_skip
        
    def call(self, x):
        identity = x
        
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.height_block(out)
        out = self.width_block(out)
        out = self.layer_block(out)
        out = self.relu(out)
        
        out = self.conv_up(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = Add()([out, identity])
        out = self.relu(out)
        
        if self.return_skip:
            return out, out
        
        else:        
            return out

