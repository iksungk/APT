from tensorflow.keras import activations, optimizers, regularizers
from tensorflow.keras.layers import Activation, Add, Dense, BatchNormalization, Concatenate, Dropout, Subtract, Flatten, Input, Lambda, Reshape
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D, UpSampling3D, ConvLSTM2D
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, MaxPool2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Layer, RepeatVector, Permute, Multiply, LeakyReLU, LayerNormalization


class SeparableConv3D(Layer):
    def __init__(self, conv_filter, kernel_size, strides, dilation_rate, use_bias, separable, reg=1e-4):
        super(SeparableConv3D, self).__init__()
        
        self.dilation_rate = dilation_rate
        self.kernel_size_xy = kernel_size
        self.kernel_size_z = kernel_size
        self.strides = strides
        self.use_bias = use_bias
        self.conv_filter = conv_filter
        self.separable = separable
        
        if self.separable == True:
            # self.kernel_size = (1,3,3), self.strides = (1,1,1) or (1,2,2)
            self.Wxy = Conv3D(conv_filter, (self.kernel_size_xy, self.kernel_size_xy, 1), strides=self.strides, dilation_rate = self.dilation_rate,
                              padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=self.use_bias, bias_regularizer=regularizers.l2(reg)) 
            self.Wz = Conv3D(conv_filter, (1, 1, self.kernel_size_z), strides=self.strides, dilation_rate = self.dilation_rate,
                             padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=self.use_bias, bias_regularizer=regularizers.l2(reg))
        
        elif self.separable == False:
            self.Wxyz = Conv3D(conv_filter, (self.kernel_size_xy, self.kernel_size_xy, self.kernel_size_xy), strides=self.strides, dilation_rate = self.dilation_rate,
                               padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=self.use_bias, bias_regularizer=regularizers.l2(reg))
            
    def call(self, p):
        if self.separable == True:
            qxy = self.Wxy(p)
            qz = self.Wz(p)
            out = Add()([qxy, qz])
            
        elif self.separable == False:
            out = self.Wxyz(p)
            
        return out


class ConvEncoderBlock(Layer):
    def __init__(self, filters, strides, separable, conv = False, reg=1e-4):
        super(ConvEncoderBlock, self).__init__()
        
        self.conv = conv
        self.relu = Activation('relu')
        
        self.conv_up1 = SeparableConv3D(conv_filter = filters[0], kernel_size = 3, strides = strides, dilation_rate = (1,1,1), 
                                        use_bias = False, separable = separable)
        self.conv_up2 = SeparableConv3D(conv_filter = filters[0], kernel_size = 3, strides = (1,1,1), dilation_rate = (1,1,1), 
                                        use_bias = False, separable = separable)
        self.conv_up3 = SeparableConv3D(conv_filter = filters[1], kernel_size = 3, strides = (1,1,1), dilation_rate = (1,1,1), 
                                        use_bias = False, separable = separable)
        self.bn_up1 = BatchNormalization()
        self.bn_up2 = BatchNormalization()
        self.bn_up3 = BatchNormalization()
        
        if conv:
            self.conv_res = SeparableConv3D(conv_filter = filters[1], kernel_size = 3, strides = strides, dilation_rate = (1,1,1), 
                                            use_bias = False, separable = separable)
            self.bn_res = BatchNormalization()
        
    def call(self, x):
        residual = x
        
        x = self.conv_up1(x)
        x = self.bn_up1(x)
        x = self.relu(x)
        x = self.conv_up2(x)
        x = self.bn_up2(x)
        x = self.relu(x)
        x = self.conv_up3(x)
        x = self.bn_up3(x)
    
        if self.conv:
            residual = self.conv_res(residual)
            residual = self.bn_res(residual)            
            
        x = Add()([x, residual])
        x = self.relu(x)
        
        return x


class ConvDecoderBlock(Layer):
    def __init__(self, filters, separable, upsample_size = (2,2,2), skip_connection = True):
        super(ConvDecoderBlock, self).__init__()
        
        self.conv_up = SeparableConv3D(conv_filter = filters, kernel_size = 3, strides = (1,1,1), dilation_rate = (1,1,1), 
                                       use_bias = False, separable = separable)
        self.conv_up2 = SeparableConv3D(conv_filter = 1, kernel_size = 3, strides = (1,1,1), dilation_rate = (1,1,1), use_bias = False, separable = separable)
        self.bn_up = BatchNormalization()
        self.bn_up2 = BatchNormalization()
        self.relu = Activation('relu')
        self.up = UpSampling3D(size = upsample_size)
        
        if skip_connection:
            self.conv_skip = SeparableConv3D(conv_filter = filters, kernel_size = 3, strides = (1,1,1), dilation_rate = (1,1,1), 
                                             use_bias = False, separable = separable)
            self.bn_skip = BatchNormalization()
        
    def call(self, x, skip):
        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x)
        x = self.up(x)
        
        if skip is None:
            x = self.conv_up2(x)
            x = self.bn_up2(x)
            x = self.relu(x)

        else:
            skip = self.conv_skip(skip)
            skip = self.bn_skip(skip)
            skip = self.relu(skip)
            x = Add()([x, skip])
        
        return x
