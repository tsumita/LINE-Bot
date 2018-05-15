#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:31:31 2017
@author: daisuke
"""
import chainer
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F 
import chainer.links as L
import numpy as np
import pickle
import datetime
import traceback

EMBED_SIZE = 300
HIDDEN_SIZE = 150
BATCH_SIZE = 40
EPOCH_NUM = 10

dictpath = "/Users/daisuke/WSL/LSTM/word_id_dict.pickle"
inputpath = "/Users/daisuke/WSL/LSTM/input/U_R_wakati.pickle"
outputpath = "/Users/daisuke/WSL/LSTM/output/S2Smodel_EMBED%s_HIDDEN%s_BATCH%s_EPOCH%s.npz"

class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__(
                xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
                eh = L.Linear(embed_size, 4*hidden_size),
                hh = L.Linear(hidden_size, 4*hidden_size)
        )
        
    def __call__(self, x, c, h):
        e = F.tanh(self.xe(x))
        return F.lstm(c, self.eh(e)+self.hh(h))
    
class LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__(
                ye = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
                eh = L.Linear(embed_size, 4*hidden_size),
                hh = L.Linear(hidden_size, 4*hidden_size),
                he = L.Linear(hidden_size, embed_size),
                ey = L.Linear(embed_size, vocab_size)
        )
        
    def __call__(self, y, c, h):
        e = F.tanh(self.ye(y))
        c, h = F.lstm(c, self.eh(e)+self.hh(h))
        t = self.ey(F.tanh(self.he(h)))
        return t, c, h
    
class Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size):
        super().__init__(
                encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
                decoder = LSTM_Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
    def encode(self, words):
        c = Variable(np.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(np.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        for w in words:
            c, h = self.encoder(w, c, h)
        self.c = c
        self.h = Variable(np.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        
    def decode(self, w):
        t, self.c, self.h = self.decoder(w, self.c, self.h)
        return t
    
    def reset(self):
        self.h = Variable(np.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.c = Variable(np.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.zerograds()
    
    def feedforward(self, enc_words, dec_words):
        batch_size = len(enc_words[0])
        self.reset()
        enc_words = [Variable(np.array(row, dtype='int32')) for row in enc_words]
        self.encode(enc_words)
        loss = Variable(np.zeros((), dtype='float32'))
        t = Variable(np.array([0 for _ in range(batch_size)], dtype='int32'))
        for w in dec_words:
            y = self.decode(t)
            t = Variable(np.array(w, dtype='int32'))
            loss += F.softmax_cross_entropy(y, t)
        return loss
    
def make_minibatch(minibatch):
    # enc_wordsの作成
    enc_words = [row[0] for row in minibatch]
    enc_max = np.max([len(row) for row in enc_words])
    # エンコーダに読み込ませる発話のリバースも行う
    enc_words = np.array([[-1]*(enc_max - len(row)) + list(reversed(row)) for row in enc_words], dtype='int32')
    enc_words = enc_words.T

    # dec_wordsの作成
    dec_words = [row[1] for row in minibatch]
    dec_max = np.max([len(row) for row in dec_words])
    dec_words = np.array([row + [-1]*(dec_max - len(row)) for row in dec_words], dtype='int32')
    dec_words = dec_words.T
    return enc_words, dec_words
      

def vocab_to_id(talk_list, vocab_dict):
    """
    語彙をIDに変換
    param talk_list: 発話と応答の組を複数集めたリスト
    param vocab_dict: 語彙に対応したIDの辞書
    return talk_list: talk_listの語彙をIDに変換したリスト
    """
    for (sentences, sentences_ind) in zip(talk_list, range(len(talk_list))):
        for (s, s_ind) in zip(sentences, range(len(sentences))):
            for (vocab, vocab_ind) in zip(s, range(len(s))):
                talk_list[sentences_ind][s_ind][vocab_ind] = vocab_dict[vocab]                
    return talk_list


def train():
    dictf = open(dictpath, 'rb')
    w_id_dict = pickle.load(dictf)
    
    vocab_size = len(w_id_dict)
    
    model = Seq2Seq(vocab_size = vocab_size,
                    embed_size=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    batch_size=BATCH_SIZE)
        
    model.reset()
    inf = open(inputpath, 'rb')
    data = pickle.load(inf)
    data = vocab_to_id(data, w_id_dict)
    
    for epoch in range(EPOCH_NUM):
        opt = optimizers.Adam()
        opt.setup(model)
        opt.add_hook(optimizer.GradientClipping(5))
        
        for num in range(len(data)//BATCH_SIZE):
            
            minibatch = data[num*BATCH_SIZE: (num+1)*BATCH_SIZE]
            enc_words, dec_words = make_minibatch(minibatch)
            
            total_loss = model.feedforward(enc_words=enc_words,
                                           dec_words=dec_words)
            
            total_loss.backward()
            opt.update()
            
        print ('Epoch %s 終了' % (epoch+1))
        outputfile = outputpath%(EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE, epoch+1)
        serializers.save_npz(outputfile, model)
        
        
if __name__ == "__main__":   
    try:
        train()
        
    except Exception:
        traceback.print_exc()
    