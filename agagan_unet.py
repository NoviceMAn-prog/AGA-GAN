import os
import time
import fnmatch
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from image_ops import get_image, save_images
import  os,shutil
import cv2
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D, Dense, \
                         Flatten, Add, PReLU, Conv2DTranspose, Lambda, UpSampling2D ,concatenate,Reshape,LeakyReLU,Multiply,Input
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from PIL import Image
from tensorflow.keras.activations import sigmoid
#tf.config.run_functions_eagerly(True)
list_file = os.path.join('data', '{0}.txt'.format('celebA'))
print(list_file)
assert os.path.exists(list_file), "no training list"
print ("Using training list: {0}".format(list_file))
with open(list_file, 'r') as f:
    data = [os.path.join( '../AAAGAN/celebA_128', l.strip()) for l in f]
assert len(data) > 0, "found 0 training data"
print ("Found {0} training images.".format(len(data)))
print(data[0])
def get_attribute():
    f = open("label_train.txt", 'r')
    labels = []
    while True:
        line = f.readline()
        if line == '':
            break
        line = line[:len(line)-1].split('\r')
        s = line[0].split()
        labels.append(s)
    labels = np.array(labels).astype(float)
    return labels
def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters):
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale=2)(x)
    x = Activation('relu')(x)
    return x


attribute_data = get_attribute()
attribute_data = attribute_data.astype(float)
class Generator(object):

    def __init__(self,lr_shape=(16,16,3),att_shape=(38,)):

        self.shape_low_reso = lr_shape
        self.attshape=att_shape

    def RDDB(self,x,nf=128,gc=64,bias=True):
        x1 = Conv2D(filters=gc, kernel_size=3, strides=1,padding='same')(x)
        x1 = LeakyReLU(alpha=0.25)(x1)
        x2_input = concatenate([x,x1],axis=-1)

        x2 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same')(x2_input)
        x2 = LeakyReLU(alpha=0.25)(x2)

        x3_input = concatenate([x,x1,x2] , axis=-1)
        x3 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same')(x3_input)
        x3 = LeakyReLU(alpha=0.25)(x3)

        x4_input = concatenate([x,x1,x2,x3] , axis=-1)
        x4 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same')(x4_input)
        x4 = LeakyReLU(alpha=0.25)(x4)

        x5_input = concatenate([x,x1,x2,x3,x4] , axis=-1)
        x5 = Conv2D(filters= nf, kernel_size=3,strides=1, padding='same')(x5_input)
        x5 = LeakyReLU(alpha=0.25)(x5)

        x5 = Lambda(lambda x: x * 0.4)(x5)
        return Add()([x5,x])


    def generator(self):
        input_layer = Input(self.shape_low_reso)
        att = Input(self.attshape)
        f_lr_1 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(input_layer)
        f_lr_1 = LeakyReLU(alpha=0.25)(f_lr_1)

        f_lr_2 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(f_lr_1)
        f_lr_2 = LeakyReLU(alpha=0.25)(f_lr_2)

        f_lr_3 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(f_lr_2)


        f_att_1 = Dense(units = 768)(att)
        f_att_1 = LeakyReLU(alpha=0.25)(f_att_1)
        f_att_1 = Reshape((16,16,3))(f_att_1)


        #reshape it to 14x12x3 figure out how
        f_att_1 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(f_att_1)
        f_att_1 = LeakyReLU(alpha=0.25)(f_att_1)

        f_att_2 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(f_att_1)
        f_att_2 = LeakyReLU(alpha=0.25)(f_att_2)

        f_att_3 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(f_att_2)
        #f_total = #concat f_att_3 and f_lr_3
        f_total = concatenate([f_att_3,f_lr_3],axis=-1)
        f1 = Conv2D(filters = 64, kernel_size = 3,padding = 'same')(f_total)
        f1 = LeakyReLU(alpha=0.25)(f1)


        f2 = Conv2D(filters = 128, kernel_size = 3,padding = 'same')(f1)
        f2 = LeakyReLU(alpha=0.25)(f2)

        f3 = Conv2D(filters = 128, kernel_size = 3,padding = 'same')(f2)
        f3 = LeakyReLU(alpha=0.25)(f3)
        print(f3.shape)

        #f4 = Conv2DTranspose(128 , kernel_size=3 , strides=(2,2),padding='same',name='check_out_the_dimension')(f3)
        #f4 = LeakyReLU(alpha=0.25)(f4)
        #print(f4.shape)
        f4_atten = upsample(f3,128)
        #f4_atten = Conv2DTranspose(filters =128 , kernel_size=3 , padding='same',strides=2)(f3)
        f4_atten = LeakyReLU(alpha=0.25)(f4_atten)





        ###########################################################################
        #MAIN BRANCH
        conv1 = Conv2D(filters= 64,kernel_size=3,padding='same')(input_layer)
        conv1 = LeakyReLU(alpha=0.25)(conv1)
        conv1 = concatenate([conv1,f1],axis=-1)

        conv2 = Conv2D(filters = 128 , kernel_size=3 , padding='same')(conv1)
        conv2 = LeakyReLU(alpha=0.25)(conv2)

        rddb1 = self.RDDB(conv2)

        conv3_input = concatenate([rddb1,f2],axis=-1)
        conv3 = Conv2D(filters = 128,kernel_size=3,padding='same')(conv3_input)
        conv3 = LeakyReLU(alpha=0.25)(conv3)

        rddb2 = self.RDDB(conv3)
        conv4_input = concatenate([rddb2,f3],axis=-1)
        conv4 = Conv2D(filters=128,kernel_size=3,padding='same')(conv4_input)
        conv4 = LeakyReLU(alpha=0.25)(conv4)

        rddb3 = self.RDDB(conv4)
        rddb3 = Lambda(lambda x: x * 0.4)(rddb3)
        final_rddb_out = Add()([rddb3,conv2])

        conv5 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1,name='findout')(final_rddb_out)
        conv5 = LeakyReLU(alpha=0.25)(conv5)
        up_conv4 = upsample(conv5,128)
        #up_conv4 = Conv2DTranspose(filters=128,kernel_size=4,padding='same',strides=2)(conv5)
        up_conv4_without = LeakyReLU(alpha=0.25)(up_conv4)
        up_conv4 = LeakyReLU(alpha=0.25)(up_conv4)
        ########################################
        #PAT blocks experiment
        pat_1_1 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(up_conv4)
        pat_1_1 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(pat_1_1)

        pat_1_2 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(f4_atten)
        pat_1_2 = Conv2D(filters=1,kernel_size=1,padding='same',strides=1)(pat_1_2)
        att_1 = sigmoid(pat_1_2)

        pat_1_1 = Multiply()([pat_1_1,att_1])
        ##here iske nhi hai neeche mc
        up_conv4 = Add()([up_conv4,pat_1_1])
        print(up_conv4.shape,f4_atten.shape)
        f4_atten = concatenate([f4_atten,up_conv4],axis=-1)
        f4_atten = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(f4_atten)
        f4_atten = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(f4_atten)



        pat_2_1 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(up_conv4)
        pat_2_1 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(pat_2_1)

        pat_2_2 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(f4_atten)
        pat_2_2 = Conv2D(filters=1,kernel_size=3,padding='same',strides=1)(pat_2_2)
        att_2 = sigmoid(pat_2_2)

        pat_2_1 = Multiply()([pat_2_1,att_2])
        up_conv4 = Add()([up_conv4,pat_2_1])
        f4_atten = concatenate([f4_atten,up_conv4])
        f4_atten = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(f4_atten)
        f4_atten = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(f4_atten)


        pat_3_1 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(up_conv4)
        pat_3_1 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(pat_3_1)

        pat_3_2 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(f4_atten)
        pat_3_2 = Conv2D(filters=128,kernel_size=3,padding='same',strides=1)(pat_3_2)
        att_3 = sigmoid(pat_3_2)

        pat_3_1 = Multiply()([pat_3_1,att_3])
        up_conv4 = Add()([up_conv4,pat_3_1])
        up_conv4 = Add()([up_conv4,up_conv4_without])




        f4 = Add()([up_conv4,f4_atten])
        f5 = upsample(f4,64)
        #f5 = Conv2DTranspose(filters = 64 , kernel_size=4 , padding='same',strides=2)(f4)
        f5 = LeakyReLU(alpha=0.25)(f5)
        f6 = upsample(f5,64)
        #f6 = Conv2DTranspose(filters = 64 , kernel_size=4 , padding='same',strides=2)(f5)
        ########################################

        up_conv3_input = concatenate([up_conv4,f4],axis=-1)
        up_conv3 = upsample(up_conv3_input,128)
        #up_conv3 = Conv2DTranspose(filters=128,kernel_size=4,padding='same',strides=2)(up_conv3_input)
        up_conv3 = LeakyReLU(alpha=0.25)(up_conv3)

        up_conv2_input = concatenate([up_conv3,f5],axis=-1)
        up_conv2 = upsample(up_conv2_input,64)
        #up_conv2 = Conv2DTranspose(filters=64,kernel_size=4,padding='same',strides=2)(up_conv2_input)
        up_conv2 = LeakyReLU(alpha=0.25)(up_conv2)

        up_conv1_input = concatenate([up_conv2,f6],axis=-1)
        #up_conv1 = upsample(up_conv1_input,64)
        up_conv1 = Conv2DTranspose(filters=64,kernel_size=3,padding='same')(up_conv1_input)
        up_conv1 = LeakyReLU(alpha=0.25)(up_conv1)


        hr = Conv2D(filters=3,kernel_size=3,padding='same',activation='tanh')(up_conv1)

        model= Model([input_layer,att],hr)
        return model

class Discriminator(object):

    def __init__(self, shape_high_reso=(112,96,3),attshape=(38,)):

        self.shape_high_reso = shape_high_reso
        self.attshape=attshape

    def discriminator(self):
        input_layer = Input(self.shape_high_reso)
        att = Input(self.attshape)
        ##########################
        #feature branch of discriminator
        f_att_1 = Dense(units = 768)(att)
        f_att_1 = LeakyReLU(alpha=0.25)(f_att_1)
        f_att_1 = Reshape((16,16,3))(f_att_1)
        #reshape it to 14x12x3 figure out how
        f_att_1 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(f_att_1)
        f_att_1 = LeakyReLU(alpha=0.25)(f_att_1)

        f_att_2 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(f_att_1)
        f_att_2 = LeakyReLU(alpha=0.25)(f_att_2)

        f_disc_input = Conv2DTranspose(filters=64,kernel_size=4,padding='same',strides=2)(f_att_2)
        f_disc_input = LeakyReLU(alpha=0.25)(f_disc_input)
        #continue the main branch of discriminator from here
        conv1 = Conv2D(filters=32 , kernel_size=3,padding='same')(input_layer)
        conv1 = LeakyReLU(alpha=0.25)(conv1)
        conv2 = Conv2D(filters=32 , kernel_size=4,padding='same',strides=2)(conv1)
        conv2 = LeakyReLU(alpha=0.25)(conv2)

        conv3 = Conv2D(filters=32 , kernel_size=3,padding='same')(conv2)
        conv3 = LeakyReLU(alpha=0.25)(conv3)
        conv4 = Conv2D(filters=64 , kernel_size=4,padding='same',strides=2)(conv3)
        conv4 = LeakyReLU(alpha=0.25)(conv4)

        conv4 = concatenate([conv4,f_disc_input],axis=-1)

        conv5 = Conv2D(filters=64 , kernel_size=3,padding='same')(conv4)
        conv5 = LeakyReLU(alpha=0.25)(conv5)
        conv6 = Conv2D(filters=128 , kernel_size=4,padding='same',strides=2)(conv5)
        conv6 = LeakyReLU(alpha=0.25)(conv6)

        conv7 = Conv2D(filters=128 , kernel_size=3,padding='same')(conv6)
        conv7 = LeakyReLU(alpha=0.25)(conv7)
        conv8 = Conv2D(filters=96 , kernel_size=4,padding='same',strides=2)(conv7)
        conv8 = LeakyReLU(alpha=0.25)(conv8)

        conv9 = Conv2D(filters=96 , kernel_size=3,padding='same')(conv8)
        conv9 = LeakyReLU(alpha=0.25)(conv9)
        #reshape conv9
        conv9 = Reshape((6144,))(conv9)
        full_1 = Dense(units=1024)(conv9)
        full_2= LeakyReLU(alpha=0.2)(full_1)
        dis_output = Dense(units = 1, activation = 'sigmoid')(full_2)
        disc_model = Model(inputs = [input_layer,att], outputs = dis_output)

        return disc_model
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D
def spatial_att_block(x,intermediate_channels):
    out = Conv2D(intermediate_channels,kernel_size=(1,1),strides=(1,1),padding='same')(x)
    #out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(1,kernel_size=(1,1),strides=(1,1),padding='same')(out)
    out = Activation('sigmoid')(out)
    return out
    
def dual_att_blocks(skip,prev,out_channels):
    up = upsample(prev,out_channels)
    #up = Conv2DTranspose(out_channels,4, strides=(2, 2), padding='same')(prev)
    #up = BatchNormalization()(up)
    up = Activation('relu')(up)
    inp_layer = Concatenate()([skip,up])
    inp_layer = Conv2D(out_channels,3,strides=(1,1),padding='same')(inp_layer)
    #inp_layer = BatchNormalization()(inp_layer)
    inp_layer = Activation('relu')(inp_layer)
    se_out = se_block(inp_layer,out_channels)
    sab = spatial_att_block(inp_layer,out_channels//4)
    #sab = Add()([sab,1])
    sab = Lambda(lambda y : y+1)(sab)
    final = Multiply()([sab,se_out])
    return final
def Attention_B(X, G, k):
    FL = int(X.shape[-1])
    init = RandomNormal(stddev=0.02)
    theta = Conv2D(k,(2,2), strides = (2,2), padding='same')(X)
    Phi = Conv2D(k, (1,1), strides =(1,1), padding='same', use_bias=True)(G)
   
    ADD = Add()([theta, Phi])
   
    #ADD = LeakyReLU(alpha=0.1)(ADD)
    ADD = Activation('relu')(ADD)
   
    #Psi = Conv2D(FL,(1,1), strides = (1,1), padding="same",kernel_initializer=init)(ADD)
    Psi = Conv2D(1,(1,1), strides = (1,1), padding="same",kernel_initializer=init)(ADD)
    Psi = Activation('sigmoid')(Psi)
    Up = Conv2DTranspose(1, (2,2), strides=(2, 2), padding='valid')(Psi)
   
    #Psi = Activation('tanh')(Psi)
   
    #Up = Conv2DTranspose(FL, (2,2), strides=(2, 2), padding='valid')(Psi)
   
    Final = Multiply()([X, Up])
    #Final = Conv2D(1, (1,1), strides = (1,1), padding="same",kernel_initializer=init)(Final)
    #Final = Conv2D(FL, (1,1), strides = (1,1), padding="same",kernel_initializer=init)(Final)
    Final = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-5)(Final)
    print(Final.shape)
    return Final
def Unet3(input_shape,n_filters,kernel=(3,3),strides=(1,1),pad='same'):
    x = input_shape
    conv1 = Conv2D(n_filters,kernel_size=kernel,strides=strides,padding=pad)(input_shape)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
   
    conv2 = Conv2D(n_filters,kernel_size=kernel,strides=strides,padding=pad)(conv1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 =  LeakyReLU(alpha=0.1)(conv2)
   
    x = Conv2D(n_filters,kernel_size = (1,1),strides = (1,1),padding = 'same')(x)
   
    return Add()([x,conv2])
def Up3(input1,input2,kernel=(3,3),stride=(1,1), pad='same'):
    #up = UpSampling2D(2)(input2)
    up = Conv2DTranspose(int(input1.shape[-1]),(1, 1), strides=(2, 2), padding='same')(input2)
    up = Concatenate()([up,input1])
    return up
    
    return Unet3(up,int(input1.shape[-1]),kernel,stride,pad)
from tensorflow.keras.layers import GlobalAveragePooling2D
def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return Multiply()([in_block, x])
class unet_Generator(object):

    def __init__(self,lr_shape=(16,16,3),att_shape=(38,),hr_shape=(128,128,6)):
        
        self.shape_low_reso = lr_shape
        self.attshape=att_shape
        self.hr_shape = hr_shape
        
    def RDDB(self,x,nf=128,gc=64,bias=True):
        x1 = Conv2D(filters=gc, kernel_size=3, strides=1,padding='same')(x)
        x1 = LeakyReLU(alpha=0.25)(x1)
        x2_input = concatenate([x,x1],axis=-1)
        
        x2 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same')(x2_input)
        x2 = LeakyReLU(alpha=0.25)(x2)
        
        x3_input = concatenate([x,x1,x2] , axis=-1)
        x3 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same')(x3_input)
        x3 = LeakyReLU(alpha=0.25)(x3)
        
        x4_input = concatenate([x,x1,x2,x3] , axis=-1)
        x4 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same')(x4_input)
        x4 = LeakyReLU(alpha=0.25)(x4)
        
        x5_input = concatenate([x,x1,x2,x3,x4] , axis=-1)
        x5 = Conv2D(filters= nf, kernel_size=3,strides=1, padding='same')(x5_input)
        x5 = LeakyReLU(alpha=0.25)(x5)
        
        x5 = Lambda(lambda x: x * 0.4)(x5)
        return Add()([x5,x])

    def unet(self):
        
        sr_layer = Input(self.hr_shape)
        x1 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(sr_layer)
        x1 = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(x1)
        print(x1.shape)
        x1 = LeakyReLU(alpha=0.25)(x1)
        x1 = se_block(x1,32)
        short1 = x1
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        x1 = Conv2D(filters = 64, kernel_size = 3,padding = 'same')(x1)
        x1 = Conv2D(filters = 64, kernel_size = 3,padding = 'same')(x1)
        x1 = LeakyReLU(alpha=0.25)(x1)
        x1 = se_block(x1,64)
        short2 = x1
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)


        x1 = Conv2D(filters = 128, kernel_size = 3,padding = 'same')(x1)
        x1 = Conv2D(filters = 128, kernel_size = 3,padding = 'same')(x1)
        x1 = LeakyReLU(alpha=0.25)(x1)
        x1 = se_block(x1,128)
        short3 = x1
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        x1 = Conv2D(filters = 256, kernel_size = 1,padding = 'same')(x1)
        x1 = Conv2D(filters = 256, kernel_size = 3,padding = 'same')(x1)
        x1 = Conv2D(filters = 256, kernel_size = 3,padding = 'same')(x1)
        x1 = LeakyReLU(alpha=0.25)(x1)
        x1 = se_block(x1,256)


        x2 = dual_att_blocks(short3,x1,128)
        #x2 = Up3(x2_preinput,x1)
        x2_conv_r = Conv2D(filters = 128, kernel_size = 3,padding = 'same')(x2)
        x2_conv = Conv2D(filters = 128, kernel_size = 3,padding = 'same')(x2_conv_r)
        x2_conv = LeakyReLU(alpha=0.25)(x2_conv)
        x2 = Add()([x2_conv_r,x2_conv])
        x2 = Conv2D(filters = 128, kernel_size = 1,padding = 'same')(x2)

        x3 = dual_att_blocks(short2,x2,64)
        #x3 = Up3(x3_preinput,x2)
        x3_conv_r = Conv2D(filters = 64, kernel_size = 3,padding = 'same')(x3)
        x3_conv = Conv2D(filters = 64, kernel_size = 3,padding = 'same')(x3_conv_r)
        x3_conv = LeakyReLU(alpha=0.25)(x3_conv)
        x3 = Add()([x3_conv_r,x3_conv])

        x4 = dual_att_blocks(short1,x3,32)
        #x4 = Up3(x4_preinput,x3)
        x4_conv_r = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(x4)
        x4_conv = Conv2D(filters = 32, kernel_size = 3,padding = 'same')(x4_conv_r)
        x4_conv = LeakyReLU(alpha=0.25)(x4_conv)
        x4 = Add()([x4_conv_r,x4_conv])
        x4 = Conv2D(filters = 3, kernel_size = 3,padding = 'same',activation='tanh')(x4)
        
        
        model= Model(inputs=sr_layer,outputs=x4)
        return model

class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return 0.25*K.mean(K.square(model(y_true) - model(y_pred))) + 0.75*mean_absolute_error(y_true,y_pred)
    
def get_optimizer():
 
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam
def get_gan_network(discriminator, shape,attshape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    gan_input_att = Input(shape=attshape)
    x = generator([gan_input,gan_input_att])
    gan_output = discriminator([x,gan_input_att])
    gan = Model(inputs=[gan_input,gan_input_att], outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 0.003],
                optimizer=optimizer)

    return gan

image_shape = (128,128,3)
X = data
X_att = attribute_data
image_shape = (128,128,3)
downscale_factor = 8
def train(epochs, batch_size,output_dir, model_save_dir):

    loss = VGG_LOSS(image_shape)

    batch_count = int(len(data) / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    att_shape = (38,)
    generator = Generator(shape,att_shape).generator()
    discriminator = Discriminator(image_shape,att_shape).discriminator()

    optimizer = get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(discriminator, shape,att_shape, generator, optimizer, loss.vgg_loss)

    loss_file = open(model_save_dir + 'losses.txt' , 'w+')
    loss_file.close()

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15,batch_size)
        #sp startpoint
        for sp in range(0,batch_count,1):
            if (sp+1)*batch_size>len(data):
                batch_end = len(data)
            else:
                batch_end = (sp+1)*batch_size
            X = data[(sp*batch_size):batch_end]
            X_att = attribute_data[(sp*batch_size):batch_end]
            hr_imgs = [get_image(sample_file,128, 128) for sample_file in X]
            hr_imgs = np.array(hr_imgs).astype(np.float32)
            lr_imgs = [get_image(sample_file,16,16) for sample_file in X]
            lr_imgs = np.array(lr_imgs).astype(np.float32)


            image_batch_hr = hr_imgs
            image_batch_lr = lr_imgs
            att = X_att[(sp*batch_size):batch_end]
            generated_images_sr = generator.predict([image_batch_lr,X_att])

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2

            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch([image_batch_hr,X_att], real_data_Y)
            d_loss_fake = discriminator.train_on_batch([generated_images_sr,X_att], fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            image_batch_hr = hr_imgs
            image_batch_lr = lr_imgs
            att = X_att
            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch([image_batch_lr,att], [image_batch_hr,gan_Y])


        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)

        loss_file = open(model_save_dir + 'losses.txt' , 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, discriminator_loss) )
        loss_file.close()
        if e % 10 == 0:
            generator.save_weights(model_save_dir + 'gen_model%d.h5' % e)
            discriminator.save_weights(model_save_dir + 'dis_model%d.h5' % e)


model_save_dir = './model_166/'
output_dir = './output/'
#train(200,100,output_dir,model_save_dir)
def train(epochs, batch_size,output_dir, model_save_dir):

    loss = VGG_LOSS(image_shape)

    batch_count = int(len(data) / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    att_shape = (38,)
    generator = Generator(shape,att_shape).generator()

    optimizer = get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    generator.load_weights('model_16/gen_model40.h5')
    unet = unet_Generator().unet()
    optimizer = get_optimizer()
    unet.compile(loss=loss.vgg_loss, optimizer=optimizer)
    unet.summary()
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15,batch_size)
        #sp startpoint
        for sp in range(0,batch_count,1):
            if (sp+1)*batch_size>len(data):
                batch_end = len(data)
            else:
                batch_end = (sp+1)*batch_size
            X = data[(sp*batch_size):batch_end]
            X_att = attribute_data[(sp*batch_size):batch_end]
            hr_imgs = [get_image(sample_file,128, 128) for sample_file in X]
            hr_imgs = np.array(hr_imgs).astype(np.float32)
            lr_imgs = [get_image(sample_file,16,16) for sample_file in X]
            bicubic_lr = []
            for i in range(len(lr_imgs)):
                b = cv2.resize(lr_imgs[i],(128,128),interpolation = cv2.INTER_CUBIC)
                bicubic_lr.append(b)
            bicubic_lr = np.array(bicubic_lr).astype(np.float32)
            lr_imgs = np.array(lr_imgs).astype(np.float32)
            

            image_batch_hr = hr_imgs
            image_batch_lr = lr_imgs
            generated_images_sr = generator.predict([image_batch_lr,X_att])

            unet_input = np.concatenate((generated_images_sr,bicubic_lr),axis=-1)
            #print(unet_input.shape)
            gan_loss = unet.train_on_batch(unet_input, image_batch_hr)


        if e % 10 == 0:
            unet.save_weights(model_save_dir + 'unetgen_model%d.h5' % e)



train(50,50,output_dir,model_save_dir)
loss = VGG_LOSS(image_shape)
downscale_factor=8
shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
att_shape = (38,)
model = Generator(shape,att_shape).generator()
optimizer = get_optimizer()
model.compile(loss=loss.vgg_loss, optimizer=optimizer)
#generator = load_model('model/gen_model150.h5' , custom_objects={'vgg_loss': loss.vgg_loss})
#model = load_model('model_16/gen_model200.h5',custom_objects={'vgg_loss': loss.vgg_loss})
model.load_weights('model_16/gen_model50.h5')
unet = unet_Generator().unet()
optimizer = get_optimizer()
unet.compile(loss=loss.vgg_loss, optimizer=optimizer)
unet.load_weights('model_166/unetgen_model20.h5')


list_file = os.path.join('Img_test.txt')
assert os.path.exists(list_file),"no testing list"
print ("Using training list: {0}".format(list_file))
with open(list_file, 'r') as f:
    data = [os.path.join('../AAAGAN/celebA_128', l.strip()) for l in f]
assert len(data) > 0, "found 0 training data"
print ("Found {0} training images.".format(len(data)))
def get_attribute():
    f = open("label_test.txt", 'r')
    labels = []
    while True:
        line = f.readline()
        if line == '':
            break
        line = line[:len(line)-1].split('\r')
        s = line[0].split()
        labels.append(s)
    labels = np.array(labels).astype(float)
    return labels
attribute_data = get_attribute()
attribute_data = attribute_data.astype(float)
X = data[:2500]
X_att = attribute_data[:2500]
hr_imgs = [get_image(sample_file,128, 128) for sample_file in X]
hr_imgs = np.array(hr_imgs).astype(np.float32)
lr_imgs = [get_image(sample_file,16,16) for sample_file in X]
lr_imgs = np.array(lr_imgs).astype(np.float32)
count=0
batch_size=50
limit=-1
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from image_similarity_measures.quality_metrics import rmse,psnr,uiq,fsim,sre
from image_similarity_measures.quality_metrics import psnr as psnr_cal
from math import log10, sqrt

def transform_255(image):
    img = (image+1.)/2.
    # save image
    im = Image.fromarray(np.uint8(img*255))
    return np.array(im)
def PSNR(original, compressed):
    original = transform_255(original)
    compressed = transform_255(compressed)
    o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    ssi = ssim(o,c)
    iss = sre(original,compressed)
    fsi = fsim(original,compressed)
    ui = uiq(original,compressed)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr,ssi,fsi,iss,ui
from image_ops import save_image
ps,ssi,fsi,iss,uiquality = 0,0,0,0,0

for sp in range(0,50):
    if (sp+1)*batch_size>len(hr_imgs):
        batch_end = len(hr_imgs)
    else:
        batch_end = (sp+1)*batch_size
    image_batch_hr = hr_imgs[(sp*batch_size):batch_end]
    batch_images_gt = image_batch_hr
    image_batch_lr = lr_imgs[(sp*batch_size):batch_end]
    att = X_att[(sp*batch_size):batch_end]
    samples = model.predict([image_batch_lr,att])
    bicubic_lr = []
    for i in range(len(image_batch_lr)):
        b = cv2.resize(image_batch_lr[i],(128,128),interpolation = cv2.INTER_CUBIC)
        bicubic_lr.append(b)
    bicubic_lr = np.array(bicubic_lr).astype(np.float32)
    unet_input = np.concatenate((samples,bicubic_lr),axis=-1)
    pred = unet.predict(unet_input)
    for i in range(50):
        print(PSNR(batch_images_gt[i],pred[i]))
        ps_i,ssi_i,fsi_i,iss_i,uiq_i =PSNR(batch_images_gt[i],pred[i])
        ps =ps + ps_i
        ssi = ssi + ssi_i
        fsi = fsi + fsi_i
        iss = iss + iss_i
        uiquality = uiquality + uiq_i
        #save_image(samples[i], os.join.path('result_attr_result', str(count) + '.jpg'))
        #save_image(batch_images_gt[i], os.join.path('result_attr_result', 'gt', str(count) + '.jpg'))
        count += 1
        limit=count
print('PSNR IS: ',ps/count)
print('SSIM is:',ssi/count)
print('FSIM is:',fsi/count)
print('SRE is:',iss/count)
print('UIQ is:',uiquality/count)

def get_region(attribute_data,data,region):
    count = 0
    testing_region = []
    testing_region_att = []
    if region == 'big_nose':
        ind = 7
    if region == 'eyeglasses':
        ind = 15
    if region == 'goatee':
        ind = 16
    if region == 'mouth_open':
        ind = 21
    if region == 'mustache':
        ind = 22
    if region == 'narrow_eyes':
        ind = 23
    if region == 'smiling':
        ind = 31
    count = 0
    for c in range(attribute_data.shape[0]):
        if attribute_data[c][ind] ==1:
            testing_region.append(data[c])
            testing_region_att.append(attribute_data[c])
            count = count+1
            if count ==1000:
                break
    testing_region_att = np.asarray(testing_region_att)
    return testing_region,testing_region_att

X,X_att = get_region(attribute_data,data,'big_nose')
print('length of big nose is:',len(X))
hr_imgs = [get_image(sample_file,128, 128) for sample_file in X]
hr_imgs = np.array(hr_imgs).astype(np.float32)
lr_imgs = [get_image(sample_file,16,16) for sample_file in X]
lr_imgs = np.array(lr_imgs).astype(np.float32)
count=0
batch_size=50
limit=-1
ps,ssi,fsi,iss,uiquality = 0,0,0,0,0
for sp in range(0,20):
    if (sp+1)*batch_size>len(hr_imgs):
        batch_end = len(hr_imgs)
    else:
        batch_end = (sp+1)*batch_size
    image_batch_hr = hr_imgs[(sp*batch_size):batch_end]
    batch_images_gt = image_batch_hr
    image_batch_lr = lr_imgs[(sp*batch_size):batch_end]
    att = X_att[(sp*batch_size):batch_end]
    samples = model.predict([image_batch_lr,att])
    bicubic_lr = []
    for i in range(len(image_batch_lr)):
        b = cv2.resize(image_batch_lr[i],(128,128),interpolation = cv2.INTER_CUBIC)
        bicubic_lr.append(b)
    bicubic_lr = np.array(bicubic_lr).astype(np.float32)
    unet_input = np.concatenate((samples,bicubic_lr),axis=-1)
    pred = unet.predict(unet_input)
    
    for i in range(50):
        ps_i,ssi_i,fsi_i,iss_i,uiq_i =PSNR(batch_images_gt[i],pred[i])
        ps =ps + ps_i
        ssi = ssi + ssi_i
        fsi = fsi + fsi_i
        iss = iss + iss_i
        uiquality = uiquality + uiq_i
        save_image(pred[i], os.path.join('local_region/big_nose', str(count) + '.jpg'))
        save_image(batch_images_gt[i], os.path.join('local_region/big_nose', 'gt', str(count) + '.jpg'))
        count += 1
        limit=count
print('big nose, PSNR IS: ',ps/count)
print('big nose, SSIM is:',ssi/count)
print('big nose, FSIM is:',fsi/count)
print('big nose, SRE is:',iss/count)
print('big nose, UIQ is:',uiquality/count)
X,X_att = get_region(attribute_data,data,'eyeglasses')
print('length of eye glasses is:',len(X))
hr_imgs = [get_image(sample_file,128, 128) for sample_file in X]
hr_imgs = np.array(hr_imgs).astype(np.float32)
lr_imgs = [get_image(sample_file,16,16) for sample_file in X]
lr_imgs = np.array(lr_imgs).astype(np.float32)
count=0
batch_size=50
limit=-1
ps,ssi,fsi,iss,uiquality = 0,0,0,0,0
for sp in range(0,10):
    if (sp+1)*batch_size>len(hr_imgs):
        batch_end = len(hr_imgs)
    else:
        batch_end = (sp+1)*batch_size
    image_batch_hr = hr_imgs[(sp*batch_size):batch_end]
    batch_images_gt = image_batch_hr
    image_batch_lr = lr_imgs[(sp*batch_size):batch_end]
    att = X_att[(sp*batch_size):batch_end]
    samples = model.predict([image_batch_lr,att])
    bicubic_lr = []
    for i in range(len(image_batch_lr)):
        b = cv2.resize(image_batch_lr[i],(128,128),interpolation = cv2.INTER_CUBIC)
        bicubic_lr.append(b)
    bicubic_lr = np.array(bicubic_lr).astype(np.float32)
    unet_input = np.concatenate((samples,bicubic_lr),axis=-1)
    pred = unet.predict(unet_input)
    for i in range(50):
        ps_i,ssi_i,fsi_i,iss_i,uiq_i =PSNR(batch_images_gt[i],pred[i])
        ps =ps + ps_i
        ssi = ssi + ssi_i
        fsi = fsi + fsi_i
        iss = iss + iss_i
        uiquality = uiquality + uiq_i
        save_image(pred[i], os.path.join('local_region/eyeglasses', str(count) + '.jpg'))
        save_image(batch_images_gt[i], os.path.join('local_region/eyeglasses', 'gt', str(count) + '.jpg'))
        count += 1
        limit=count
print('eyeglasses, PSNR IS: ',ps/count)
print('eyeglasses, SSIM is:',ssi/count)
print('eyeglasses, FSIM is:',fsi/count)
print('eyeglasses, SRE is:',iss/count)
print('eyeglasses, UIQ is:',uiquality/count)

X,X_att = get_region(attribute_data,data,'goatee')
print('length of goatee is:',len(X))
hr_imgs = [get_image(sample_file,128, 128) for sample_file in X]
hr_imgs = np.array(hr_imgs).astype(np.float32)
lr_imgs = [get_image(sample_file,16,16) for sample_file in X]
lr_imgs = np.array(lr_imgs).astype(np.float32)
count=0
batch_size=50
limit=-1
ps,ssi,fsi,iss,uiquality = 0,0,0,0,0
for sp in range(0,10):
    if (sp+1)*batch_size>len(hr_imgs):
        batch_end = len(hr_imgs)
    else:
        batch_end = (sp+1)*batch_size
    image_batch_hr = hr_imgs[(sp*batch_size):batch_end]
    batch_images_gt = image_batch_hr
    image_batch_lr = lr_imgs[(sp*batch_size):batch_end]
    att = X_att[(sp*batch_size):batch_end]
    samples = model.predict([image_batch_lr,att])
    bicubic_lr = []
    for i in range(len(image_batch_lr)):
        b = cv2.resize(image_batch_lr[i],(128,128),interpolation = cv2.INTER_CUBIC)
        bicubic_lr.append(b)
    bicubic_lr = np.array(bicubic_lr).astype(np.float32)
    unet_input = np.concatenate((samples,bicubic_lr),axis=-1)
    pred = unet.predict(unet_input)
    for i in range(50):
        ps_i,ssi_i,fsi_i,iss_i,uiq_i =PSNR(batch_images_gt[i],pred[i])
        ps =ps + ps_i
        ssi = ssi + ssi_i
        fsi = fsi + fsi_i
        iss = iss + iss_i
        uiquality = uiquality + uiq_i
        save_image(pred[i], os.path.join('local_region/goatee', str(count) + '.jpg'))
        save_image(batch_images_gt[i], os.path.join('local_region/goatee', 'gt', str(count) + '.jpg'))
        count += 1
        limit=count
print('goatee, PSNR IS: ',ps/count)
print('goatee, SSIM is:',ssi/count)
print('goatee, FSIM is:',fsi/count)
print('goatee, SRE is:',iss/count)
print('goatee, UIQ is:',uiquality/count)

X,X_att = get_region(attribute_data,data,'mouth_open')
print('length of mouth_open is:',len(X))
hr_imgs = [get_image(sample_file,128, 128) for sample_file in X]
hr_imgs = np.array(hr_imgs).astype(np.float32)
lr_imgs = [get_image(sample_file,16,16) for sample_file in X]
lr_imgs = np.array(lr_imgs).astype(np.float32)
count=0
batch_size=50
limit=-1
ps,ssi,fsi,iss,uiquality = 0,0,0,0,0
for sp in range(0,20):
    if (sp+1)*batch_size>len(hr_imgs):
        batch_end = len(hr_imgs)
    else:
        batch_end = (sp+1)*batch_size
    image_batch_hr = hr_imgs[(sp*batch_size):batch_end]
    batch_images_gt = image_batch_hr
    image_batch_lr = lr_imgs[(sp*batch_size):batch_end]
    att = X_att[(sp*batch_size):batch_end]
    samples = model.predict([image_batch_lr,att])
    bicubic_lr = []
    for i in range(len(image_batch_lr)):
        b = cv2.resize(image_batch_lr[i],(128,128),interpolation = cv2.INTER_CUBIC)
        bicubic_lr.append(b)
    bicubic_lr = np.array(bicubic_lr).astype(np.float32)
    unet_input = np.concatenate((samples,bicubic_lr),axis=-1)
    pred = unet.predict(unet_input)
    for i in range(50):
        ps_i,ssi_i,fsi_i,iss_i,uiq_i =PSNR(batch_images_gt[i],pred[i])
        ps =ps + ps_i
        ssi = ssi + ssi_i
        fsi = fsi + fsi_i
        iss = iss + iss_i
        uiquality = uiquality + uiq_i
        save_image(pred[i], os.path.join('local_region/mouth_open', str(count) + '.jpg'))
        save_image(batch_images_gt[i], os.path.join('local_region/mouth_open', 'gt', str(count) + '.jpg'))
        count += 1
        limit=count
print('mouth_open, PSNR IS: ',ps/count)
print('mouth_open, SSIM is:',ssi/count)
print('mouth_open, FSIM is:',fsi/count)
print('mouth_open, SRE is:',iss/count)
print('mouth_open, UIQ is:',uiquality/count)

X,X_att = get_region(attribute_data,data,'mustache')
print('length of mustache is:',len(X))
hr_imgs = [get_image(sample_file,128, 128) for sample_file in X]
hr_imgs = np.array(hr_imgs).astype(np.float32)
lr_imgs = [get_image(sample_file,16,16) for sample_file in X]
lr_imgs = np.array(lr_imgs).astype(np.float32)
count=0
batch_size=50
limit=-1
ps,ssi,fsi,iss,uiquality = 0,0,0,0,0
for sp in range(0,8):
    if (sp+1)*batch_size>len(hr_imgs):
        batch_end = len(hr_imgs)
    else:
        batch_end = (sp+1)*batch_size
    image_batch_hr = hr_imgs[(sp*batch_size):batch_end]
    batch_images_gt = image_batch_hr
    image_batch_lr = lr_imgs[(sp*batch_size):batch_end]
    att = X_att[(sp*batch_size):batch_end]
    samples = model.predict([image_batch_lr,att])
    bicubic_lr = []
    for i in range(len(image_batch_lr)):
        b = cv2.resize(image_batch_lr[i],(128,128),interpolation = cv2.INTER_CUBIC)
        bicubic_lr.append(b)
    bicubic_lr = np.array(bicubic_lr).astype(np.float32)
    unet_input = np.concatenate((samples,bicubic_lr),axis=-1)
    pred = unet.predict(unet_input)
    for i in range(50):
        ps_i,ssi_i,fsi_i,iss_i,uiq_i =PSNR(batch_images_gt[i],pred[i])
        ps =ps + ps_i
        ssi = ssi + ssi_i
        fsi = fsi + fsi_i
        iss = iss + iss_i
        uiquality = uiquality + uiq_i
        save_image(pred[i], os.path.join('local_region/mustache', str(count) + '.jpg'))
        save_image(batch_images_gt[i], os.path.join('local_region/mustache', 'gt', str(count) + '.jpg'))
        count += 1
        limit=count
print('mustache, PSNR IS: ',ps/count)
print('mustache, SSIM is:',ssi/count)
print('mustache, FSIM is:',fsi/count)
print('mustache, SRE is:',iss/count)
print('mustache, UIQ is:',uiquality/count)

X,X_att = get_region(attribute_data,data,'narrow_eyes')
print('length of narrow_eyes is:',len(X))
hr_imgs = [get_image(sample_file,128, 128) for sample_file in X]
hr_imgs = np.array(hr_imgs).astype(np.float32)
lr_imgs = [get_image(sample_file,16,16) for sample_file in X]
lr_imgs = np.array(lr_imgs).astype(np.float32)
count=0
batch_size=50
limit=-1
ps,ssi,fsi,iss,uiquality = 0,0,0,0,0
for sp in range(0,20):
    if (sp+1)*batch_size>len(hr_imgs):
        batch_end = len(hr_imgs)
    else:
        batch_end = (sp+1)*batch_size
    image_batch_hr = hr_imgs[(sp*batch_size):batch_end]
    batch_images_gt = image_batch_hr
    image_batch_lr = lr_imgs[(sp*batch_size):batch_end]
    att = X_att[(sp*batch_size):batch_end]
    samples = model.predict([image_batch_lr,att])
    bicubic_lr = []
    for i in range(len(image_batch_lr)):
        b = cv2.resize(image_batch_lr[i],(128,128),interpolation = cv2.INTER_CUBIC)
        bicubic_lr.append(b)
    bicubic_lr = np.array(bicubic_lr).astype(np.float32)
    unet_input = np.concatenate((samples,bicubic_lr),axis=-1)
    pred = unet.predict(unet_input)
    for i in range(50):
        ps_i,ssi_i,fsi_i,iss_i,uiq_i =PSNR(batch_images_gt[i],pred[i])
        ps =ps + ps_i
        ssi = ssi + ssi_i
        fsi = fsi + fsi_i
        iss = iss + iss_i
        uiquality = uiquality + uiq_i
        save_image(pred[i], os.path.join('local_region/narrow_eyes', str(count) + '.jpg'))
        save_image(batch_images_gt[i], os.path.join('local_region/narrow_eyes', 'gt', str(count) + '.jpg'))
        count += 1
        limit=count
print('narrow_eyes, PSNR IS: ',ps/count)
print('narrow_eyes, SSIM is:',ssi/count)
print('narrow_eyes, FSIM is:',fsi/count)
print('narrow_eyes, SRE is:',iss/count)
print('narrow_eyes, UIQ is:',uiquality/count)


