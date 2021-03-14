import tenorflow as tf
import numpy as np

class EmbLayer(tf.keras.Model):
    def __init__(self,vocab_size,hidden_size,seg=True):
        super(EmbLayer,self).__init__()
        self.enc_emb = tf.keras.layers.Embedding(vocab_size+1,hidden_size) # vocab_size + 1 for UNK Token
        #self.positional_emb = 
        #self.seg = seg
        #self.sep_id = tokenizer.sep_token_id
        #self.segement_emg =


    def call(self,token):
        enc_emb = self.enc_emb(token)

        return enc_emb 

def emb_seg(tok,sep_token_id): # tok는 그냥 리스트 , seg_emb 는 ndarray
    idx = tok.index(sep_token_id)
    seg_emb = np.ones(len(tok))
    seg_emb[idx+1:] +=1 

    return seg_emb
