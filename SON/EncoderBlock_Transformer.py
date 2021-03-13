import tensorflow as tf
import numpy as np 

config = {'hidden_size' : 768,
          'max_seq' :256,
          'head_num' : 12,
          'dropout': 0.1,
          'layer_norm_epsilon': 1e-12}

class EncoderBlock(tf.keras.Model):
    def __init__(self,hidden_size,head_num,dropout,layer_norm_epsilon):
        super(EncoderBlock,self).__init__()

        self.MultiHeadAttention = MultiHeadAttention(hidden_size,head_num)
        self.MHA_Dropout = tf.keras.layers.Dropout(dropout)
        self.MHA_Normalization = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)

        self.FFN = tf.keras.Sequential([
                                        tf.keras.layers.Dense(hidden_size*4),
                                        tf.keras.layers.LeakyReLU(),
                                        tf.keras.layers.Dense(hidden_size)])
        self.FFN_Dropout = tf.keras.layers.Dropout(dropout)
        self.FFN_Normalization = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
    
    def call(self,x):
        normalized_x = self.MHA_Normalization(x)
        attention_weight , attention_output = self.MultiHeadAttention(normalized_x,normalized_x,normalized_x)
        attention_output = x + self.MHA_Dropout(attention_output)

        normalized_attention_output = self.FFN_Normalization(attention_output)
        FFN_output = attention_output + self.FFN_Dropout(self.FFN(normalized_attention_output))

        return attention_weight, FFN_output

def testEncoder(config):
    sample_tensor = tf.random.normal([10,config['max_seq'],config['hidden_size']])
    Encoder = EncoderBlock(config['hidden_size'],config['head_num'],config['dropout'],config['layer_norm_epsilon'])
    attention_weight , encoder_output = Encoder(sample_tensor)

    if encoder_output.shape == sample_tensor.shape:
        print('Valid!')
    else:
        print('Shape Error')
