import numpy as np
import math
from tensorflow.keras.utils import Sequence
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

'''
input:
type(data_list) = list 
data_list['무작위 문장1.무작위 문장2.','무작위 문장3. 무작위 문장4.' ...]

output:
list[list[tensor]]
'''

# dataloader = {'embeddding': [...],'segment':[...],'positional_encoding':[...]}

class Dataloader(Sequence):
    def __init__(self, data_list, batch_size, shuffle=False):
        self.data_list = \
        tokenizer.batch_encode_plus(data_list, max_length=256, truncation=True, padding=True, return_tensors='tf')[
            'input_ids']
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.data_list) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        data_batch = [self.data_list[i] for i in indices]

        seg_list = []
        for batch in self.data_list:
            for sequence in batch:
                try:
                    idx = sequence.index(119) +1
                    seg_a = np.zeros(idx)
                    seg_b = np.ones(len(sequence)-idx)
                    seg = np.concatenate((seg_a,seg_b))

                except:
                    seg = np.zeros(len(sequence))
                seg_list.appen(seg)
        seg_batch = [seg_list[i] for i in indices]
        return {'token':data_batch,'segment':seg_batch}

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indices)