#!/usr/bin/env python
# coding: utf-8

import keras
from keras.applications import resnet50
from keras.layers import Conv2D,BatchNormalization,Activation,Add,Concatenate,ZeroPadding2D
from keras.layers import MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D,GlobalMaxPool2D
from keras.layers import Input,Flatten,Dense
from keras.models import Model
import cv2
import numpy as np

def indentity_block(input_tensor,kernel_size,filters,stage,block):
    
    filter1 , filter2 , filter3 = filters
    
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block +'_branch'
    
    x = Conv2D(filter1,(1,1),name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(axis=3,name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filter2,(3,3),padding='same',name=conv_name_base+'2b')(x)
    x = BatchNormalization(axis=3,name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filter3,(1,1),name=conv_name_base+'2c')(x)
    x = BatchNormalization(axis=3,name=bn_name_base+'2c')(x)
    
    x = Add()([x , input_tensor])
    x = Activation('relu')(x)
    
    return x


def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):
    
    filter1,filter2,filter3 = filters
    
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'
    
    x = Conv2D(filter1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(axis=3,name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filter2,(3,3),padding='same',name=conv_name_base+'2b')(x)
    x = BatchNormalization(axis=3,name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filter3,(1,1),name=conv_name_base+'2c')(x)
    x = BatchNormalization(axis=3,name=bn_name_base+'2c')(x)
    
    
    shortcut = Conv2D(filter3,(1,1),strides=strides,name=conv_name_base+'1')(input_tensor)
    shortcut = BatchNormalization(axis=3,name=bn_name_base+'1')(shortcut)
    
    x = Add()([x,shortcut])
    
    x = Activation('relu')(x)
    
    return x




def ResNet50_v1(include_top = False):
    
    input_tensor = Input(shape=(224,224,3))

    x = Conv2D(64,(7,7),padding='same',strides=(2,2),name='conv1')(input_tensor)
    x = BatchNormalization(axis=3,name='bn_conv1')(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((3,3),strides=(2,2))(x)
    x = conv_block(x,3,[64,64,256],2,'a',strides=(1,1))
    x = indentity_block(x,3,[64,64,256],2,'b')
    x = indentity_block(x,3,[64,64,256],2,'c')
    
    x = conv_block(x,3,[128,128,512],3,'a')
    x = indentity_block(x,3,[128,128,512],3,'b')
    x = indentity_block(x,3,[128,128,512],3,'c')
    x = indentity_block(x,3,[128,128,512],3,'d')
    
    x = conv_block(x,3,[256,256,1024],4,'a')
    x = indentity_block(x,3,[256,256,1024],4,'b')
    x = indentity_block(x,3,[256,256,1024],4,'c')
    x = indentity_block(x,3,[256,256,1024],4,'d')
    x = indentity_block(x,3,[256,256,1024],4,'e')
    x = indentity_block(x,3,[256,256,1024],4,'f')
    
    x = conv_block(x,3,[512,512,2048],5,'a')
    x = indentity_block(x,3,[512,512,2048],5,'b')
    x = indentity_block(x,3,[512,512,2048],5,'c')
    
    x = AveragePooling2D((7,7))(x)
    
    if include_top:
        x = Flatten()(x)
        x = Dense(1000,activation='softmax',name="fc1000")(x)
    
    model = Model(input_tensor,x)
    return model




def indentity_block_v2(input_tensor,kernel_size,filters,stage,block):
    
    filter1, filter2, filter3 = filters
    
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'
    
    x = BatchNormalization(axis=3,name=bn_name_base+'2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filter1,(1,1),name=conv_name_base+'2a')(x)
    
    x = BatchNormalization(axis=3,name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2,(3,3),padding='same',name=conv_name_base+'2b')(x)
    
    x = BatchNormalization(axis=3,name=bn_name_base+'2c')(x)
    x = Activation('relu')(x)
    x = Conv2D(filter3,(1,1),name=conv_name_base+'2c')(x)
    
    
    
    x = Add()([x,input_tensor])
    
    return x
    
    


def conv_block_v2(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):
    
    filter1 , filter2 , filter3 = filters
    
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'
    
    if strides[0] == 1 :
        x = Conv2D(filter1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
    else :
        x = BatchNormalization(axis=3,name=bn_name_base+'2a')(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filter1,(1,1),strides=strides,name=conv_name_base+'2a')(x)
    
    x = BatchNormalization(axis=3,name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2,(3,3),padding='same',name=conv_name_base+'2b')(x)
    
    x = BatchNormalization(axis=3,name=bn_name_base+'2c')(x)
    x = Activation('relu')(x)
    x = Conv2D(filter3,(1,1),name=conv_name_base+'2c')(x)
    
    shortcut = Conv2D(filter3,(1,1),strides=strides,name=conv_name_base+'1')(input_tensor)
    
    
    x = Add()([x,shortcut])
    
    return x



def ResNet50_v2(include_top = False):
    
    input_tensor = Input(shape=(224,224,3))
    
    x = Conv2D(64,(7,7),padding='same',strides=(2,2),name='conv1')(input_tensor)
    x = BatchNormalization(axis=3,name='bn_conv1')(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((3,3),strides=(2,2))(x)
    x = conv_block_v2(x,3,[64,64,256],2,'a',strides=(1,1))
    x = indentity_block_v2(x,3,[64,64,256],2,'b')
    x = indentity_block_v2(x,3,[64,64,256],2,'c')
     
    x = conv_block_v2(x,3,[128,128,512],3,'a')
    x = indentity_block_v2(x,3,[128,128,512],3,'b')
    x = indentity_block_v2(x,3,[128,128,512],3,'c')
    x = indentity_block_v2(x,3,[128,128,512],3,'d')
    
    x = conv_block_v2(x,3,[256,256,1024],4,'a')
    x = indentity_block_v2(x,3,[256,256,1024],4,'b')
    x = indentity_block_v2(x,3,[256,256,1024],4,'c')
    x = indentity_block_v2(x,3,[256,256,1024],4,'d')
    x = indentity_block_v2(x,3,[256,256,1024],4,'e')
    x = indentity_block_v2(x,3,[256,256,1024],4,'f')
    
    x = conv_block_v2(x,3,[512,512,2048],5,'a')
    x = indentity_block_v2(x,3,[512,512,2048],5,'b')
    x = indentity_block_v2(x,3,[512,512,2048],5,'c')
    
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = AveragePooling2D((7,7),name='avg_pool')(x)
    
    if include_top :
        x = Flatten()(x)
        x = Dense(1000,activation='softmax',name='fc1000')(x)
        
    model = Model(input_tensor,x)
    
    return model


if __name__ == '__main__':
    
    
    model_v1 = ResNet50_v1(True)
    model_v2 = ResNet50_v2(True)

    img = cv2.imread('./car3.jpg')
    img = cv2.resize(img,(224,224))
    img = resnet50.preprocess_input(img)

    model_v1.load_weights('./resnet50_weights_tf_dim_ordering_tf_kernels.h5',by_name=True)
    preds = model_v1.predict(np.expand_dims(img, axis=0))


