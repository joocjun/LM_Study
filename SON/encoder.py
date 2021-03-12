import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention

class EncoderBlock(tf.keras.Model):
    def __init__(self,hidden_size,head):
        super(EncoderBlock,self).__init__()
        
        self.attention = MultiHeadAttention(hidden_size,head)
        self.attention_norm = tf.keras.layers.BatchNormalization()

        self.FFN = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size*4),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(hidden_size)])
        
        self.FFN_norm = tf.keras.layers.BatchNormalization()

    def call(self,x): # x = q = k = v
        z = self.attention_norm(x)
        attention_weight, output = self.attention(x,x,x)
        z= x + output
        z = z + self.FFN(self.FFN_norm(z))

        return z

def test_encoder(hidden_size=768.,head_num=12):
    sample_tensor = tf.random.normal([10,256,768])
    ENCODER = EncoderBlock(hidden_size,head_num)
    output = ENCODER(sample_tensor)
    
    if output.shape == sample_tensor.shape:
        print('valid')
        

if __name__ == "__main__":
    test_encoder()
