import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Dropout, Input, Lambda, TimeDistributed, Dense
import numpy as np
import os

def build_model(batch_size,
                seq_len=512,
                embed_dim=768,
                pos_num=512,
                transformer_layer_num=12,
                attention_head_num=12,
                feed_forward_dim=3072,
                dropout_rate=0.1,
                attention_activation=None,
                feed_forward_activation='gelu',
                training=True,
                trainable=None):

    return

def pretrain_model():

    def compile_model():
        
        return
    


    return

def finetune_model():


    return
