from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import imgTextDataset
from trainer import condGANTrainer as trainer

import re
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

from textembeding import Model
from tensorboardX import SummaryWriter

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/eval_bird.yml', type=str)  # eval_bird.yml  bird_attn2.yml
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example(text_encoder, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().encode('utf8').decode('utf-8').split('\n')
        captions = []
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r", encoding='UTF-8') as f:
                print('Load from:', name)
                # sentences = f.read().encode('utf8').decode('utf-8').split('\n')
                # a list of indices for a sentence
                captions = []
                captionss = []
                cap_lens = []
                while 1:
                    line = f.readline()
                    if not line:
                        f.close()
                        break
                    captions.append(line)
                for sent in captions:
                    if len(sent) == 0:
                        continue
                    sent = re.sub('[,.]', ' ', sent)
                    sent = sent.replace("\ufffd\ufffd", " ")
                    # tokenizer = RegexpTokenizer(r'\w+')
                    tokens = text_encoder.tensor_tokens_id(sent)
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue
                    num_words = len(tokens)
                    sent_caption = np.asarray(tokens).astype('int64')
                    x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
                    if num_words <= cfg.TEXT.WORDS_NUM:
                        x[:num_words, 0] = sent_caption
                    else:
                        ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                        np.random.shuffle(ix)
                        ix = ix[:cfg.TEXT.WORDS_NUM]
                        ix = np.sort(ix)
                        x[:, 0] = sent_caption[ix]
                        x_len = cfg.TEXT.WORDS_NUM

                    # rev = []
                    # for t in tokens:
                    #     t = t.encode('ascii', 'ignore').decode('ascii')
                    #     if len(t) > 0 and t in wordtoix:
                    #         rev.append(wordtoix[t])
                    xx = torch.from_numpy(x)
                    captionss.append(xx)
                    cap_lens.append(len(xx))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captionss), max_len), dtype='int64')
            for i in range(len(captionss)):
                idx = sorted_indices[i]
                cap = captionss[idx]
                cap = cap.squeeze()
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic ,text_encoder)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    text_encoder = Model()  # cfg.TRAIN.NET_E
    state_dict = \
        torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage) # 导入文本编码器的参数
    text_encoder.load_state_dict(state_dict)
    # print('Load ', cfg.TRAIN.NET_E)
    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder.eval()
    dataset = imgTextDataset(cfg.DATA_DIR, text_encoder, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    # assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(text_encoder, output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(text_encoder, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
