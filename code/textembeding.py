from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import shutil

import torch.utils.data as data
from PIL import Image
import PIL
import os.path
import pickle
import random
import h5py
import numpy as np
import pandas as pd

from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import re
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)  # 返回隐层bert-base-multilingual-cased  bert-base-chinese
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')   # distilbert-base-uncased
        print("bert")
    def embd(self, tokens_id):
        # tokens = self.tokenizer.tokenize(inputs)
        # # print(tokens)
        # tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        # print(tokens_id)   sourceTensor.clone().detach()
        # tokens_id_tensor = tokens_id.clone().detach()
        # tokens_id_tensor = tokens_id.numpy().tolist()
        # tokens_id_tensor = tokens_id_tensor.cuda()

        tokens_id_tensor = tokens_id.unsqueeze(0)  # torch.tensor(tokens_id).unsqueeze(0)
        outputs = self.embedder(tokens_id_tensor)
        # print(outputs[0])
        # print("out[1]:",outputs[1])
        # vec = np.zeros(768, np.float32)
        sentence = outputs[1]
        sentence = sentence.squeeze(0)

        words = outputs[0]
        words = words.squeeze(0)
        wordsm = words.t()
        return sentence, wordsm
    def encode(self,input):
        # tokens = self.tokenizer.convert_ids_to_tokens(input)
        # encoded_input = self.tokenizer(input, return_tensors='pt')
        input = input.unsqueeze(0)
        output = self.embedder(input)
        return output
    def tensor_tokens_id(self,inputs):
        tokens = self.tokenizer.tokenize(inputs)  #分词
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        # tokens_id = np.array(tokens_id)
        # tokens_id = torch.from_numpy(tokens_id)
        return tokens_id
    def id_tensor_tokens(self,tokens_id):
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_id)
        return tokens

    def forward(self, captions):
        i = 0
        sent_emb = np.zeros((30, 768), dtype='float32')
        sent_emb = torch.as_tensor(torch.from_numpy(sent_emb), dtype=torch.float32)
        words_emb = np.zeros((30, 768, 18), dtype='float32')
        words_emb = torch.as_tensor(torch.from_numpy(words_emb), dtype=torch.float32)
        sent_emb = sent_emb.cuda()
        words_emb = words_emb.cuda()
        while (i < 30):
            captions1 = captions[i].cpu()
            captions1 = captions1.numpy().tolist()
            captions1 = torch.LongTensor(captions1)
            # print(captions1.max())
            # print(captions1.min())
            captions1 = captions1.cuda()
            sent_emb[i], words_emb[i] = self.embd(captions1)
            i = i + 1
        # sent_emb = sent_emb.cuda()
        # words_emb = words_emb.cuda()
        return sent_emb,words_emb
# 打印读取文件目录 图片从单独文件夹中取出
# if __name__ == "__main__":
    # textfilenames = []
    # imagefilenames = []
    # newimagefilenames = []
    # newtextfilenames = []
    # path = '/root/hk/AttnGAN-master-cloth/cloth/clothdata/1.txt'
    # path1 = '/root/hk/AttnGAN-master-cloth/cloth/clothdata/2.txt'
    # data_dir = 'out/qun-textcn/'
    # count = 0
    # model = Model()
    # model = model.cuda()
    # model.train()
    # textpath = os.path.join('/root/hk/AttnGAN-master/data/birds/text/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.txt')
    # try:
    #     with open(textpath, 'r', encoding='UTF-8') as f:  # 打开新的文本
    #         key = f.readline()  # 读取文本数据
    # except:
    #     with open(textpath, 'r', encoding='gbk') as f:  # 打开新的文本
    #         key = f.readline()  # 读取文本数据
    # # key = re.sub('[A-Za-z0-9\!\%\[\]\,\。\【\】]', '', key)
    # # vec = model.embd(key)
    #
    # # vec = model.tokenizer(key, return_tensors='pt')
    # # vec = vec.cuda()
    # # sent_emb, words_emb = model.embd(**vec)
    #
    # vec = key.replace("\ufffd\ufffd", " ")
    # vec = model.tensor_tokens_id(key.lower())
    # # vec = vec.squeeze(0)
    #
    # vec = torch.LongTensor(vec)
    # vec = vec.cuda()
    # sent_emb,words_emb = model.embd(vec)
    # # vec = model.id_tensor_tokens(vec)
    #
    # # vec = vec.squeeze(0)
    # # fpath = os.path.join('D:\\HeDataSet\\pytorchtest\\StackGAN-Pytorch\\embeding', path)
    # words_emb = np.array(words_emb.detach().numpy())
    # np.savetxt(path, words_emb)
    # sent_emb = np.array(sent_emb.detach().numpy())
    # np.savetxt(path1, sent_emb)
    # textemd = np.loadtxt(path).astype(np.float64)
    # textemding = torch.from_numpy(textemd)
    # print(textemding.shape)
