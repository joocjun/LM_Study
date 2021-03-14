import tensorflow as tf
from EncoderBlock_Transformer import EncoderBlock
from MultiHeadAttention import MultiHeadAttention
from Encoding import EmbLayer

class Bert(tf.keras.Model):
  def __init__(self,):
    super(Bert,self).__init__()
    
    self.emb = EmbLayer()
    self.enc_layers = [EncoderBlock(hidden_size, head_num, 0.1, 1e-12) 
                       for _ in range(6)]
  def call(self):
    
    
    
    
