import tensorflow as tf

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
