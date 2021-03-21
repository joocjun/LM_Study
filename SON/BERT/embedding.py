import numpy as np
import tensorflow as tf
# BertTokenizer ('bert-base-cased') 기준 '.' == 119

class EmbeddingLayer(tf.keras.Model):
    def __init__(self, data, hidden_size, max_length):
        super(EmbeddingLayer, self).__init__()
        self.data = data  # 리스트 안에 리스트 구
        self.hidden_size = hidden_size
        self.embedding = tf.keras.layers.Dense(hidden_size)
        self.positional_embedding = tf.keras.layers.Dense(hidden_size)
        self.segment_embedding = tf.keras.layers.Dense(hidden_size)

    def __call__(self):  # type(data) == list조
        embedded_batch = []
        n = 0
        for sequence in self.data:
            tokenized_batch = tf.constant(sequence['token'])
            segment_batch = tf.constant(sequence['segment'])

            tokenized_batch = tf.reshape(tokenized_batch, [500, 256, 1])
            segment_batch = tf.reshape(segment_batch, [500, 256, 1])

            basic_embed = self.embedding(tokenized_batch)
            print(basic_embed.shape)
            seg_embed = self.segment_embedding(segment_batch)
            n += 1
            print(n)

            embed = basic_embed + seg_embed
            embed = tf.add(embed, self.positional_encoding(basic_embed.shape[1], self.hidden_size))
            embedded_batch.append(embed)
        return embedded_batch

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, max_length, hidden_size):
        angle_rads = self.get_angles(np.arange(max_length)[:, np.newaxis], np.arange(hidden_size)[np.newaxis, :],
                                     hidden_size)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)








