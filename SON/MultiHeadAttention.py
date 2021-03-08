import tensorflow as tf
import numpy as np


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, hidden_size, head, masked=False):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.head = head  # head의 수

        self.wq = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.wk = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.wv = tf.keras.layers.Dense(hidden_size, use_bias=False)

        self.linear = tf.keras.layers.Dense(hidden_size, use_bias=False)

        self.scale = tf.keras.layers.Lambda(lambda x: x / np.sqrt(hidden_size))
        self.masked = masked

    def call(self, q, k, v, mask=None):
        """
        wq = tf.split(self.wq(q), num_or_size_split = self.head, axis = -1)
        wk = tf.split(self.wk(k), num_or_size_split = self.head, axis = -1)
        wv = tf.split(self.wv(v), num_or_size_split = self.head, axis = -1)
        # (bs,ts,hs) * head

        wq_concat = tf.concat(wq,axis=0)
        wk_concat = tf.concat(wk,axis=0)
        wv_concat = tf.concat(wv,axis=0)

        # (bs*head,ts,hs)
        """
        assert q.shape[-1] % self.head == 0

        wq = tf.reshape(self.wq(q), [self.head, q.shape[0], q.shape[1], -1])
        wk = tf.reshape(self.wk(k), [self.head, k.shape[0], k.shape[1], -1])
        wv = tf.reshape(self.wv(v), [self.head, v.shape[0], v.shape[1], -1])
        # (head_n,bs,ts,hs/head_n)
        scaled_attention_logit = self.scale(tf.matmul(wq, wk, transpose_b=True))

        if self.masked:
            mask = (1 - tf.linalg.band_part(tf.ones(scaled_attention_logit.shape[1:]), -1, 0)) * -1e9
            scaled_attention_logit = tf.reshape([head + mask for head in scaled_attention_logit],
                                                 scaled_attention_logit.shape)

        attention_weight = tf.nn.softmax(scaled_attention_logit, axis=-1)
        # head_n,bs,ts,hs/head_n
        output = tf.reshape(tf.matmul(attention_weight, wv), q.shape)
        output = self.linear(output)
        # (bs,ts,hs)

        return attention_weight, output
