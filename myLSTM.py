from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F 
import chainer.links as L
import numpy as np
import pickle
import MeCab
import traceback
from LSTM_s2s import LSTM_Encoder, LSTM_Decoder, Seq2Seq


class myLSTMunit():
	def __init__(self):
		self.enbed_size = 300
		self.hidden_size = 150
		self.train_batch_size = 40
		self.test_size = 1
		self.__dictpath = "./LSTM/word_id_dict.pickle"
		self.__inputpath = "./LSTM/output/S2Smodel_EMBED%s_HIDDEN%s_BATCH%s_EPOCH%s.npz"

	def test(self, inputw, epoch=10):
	    
	    dictf = open(self.__dictpath, 'rb')
	    w_id_dict = pickle.load(dictf)
	    id_w_dict = {v:k for k, v in w_id_dict.items()}
	    
	    vocab_size = len(w_id_dict)
	    
	    model = Seq2Seq(vocab_size = vocab_size,
	                    embed_size=self.enbed_size,
	                    hidden_size=self.hidden_size,
	                    batch_size=self.test_size)
	    
	    inputfile = self.__inputpath%(self.enbed_size, self.hidden_size, self.train_batch_size, epoch)
	    serializers.load_npz(inputfile, model)
	    
	    mt = MeCab.Tagger("-Ochasen")
	    mt.parse('')
	    node = mt.parseToNode(inputw)
	    inputid_list = []
	    while node:
	        if node.surface == '':
	            node = node.next
	            continue
	        inputid_list.insert(0,w_id_dict[node.surface])
	        node = node.next
	    
	    enc_words = [Variable(np.array([row], dtype='int32')) for row in inputid_list]
	    model.encode(enc_words)
	    t = Variable(np.array([0], dtype='int32'))
	    
	    count = 0
	    talk = ""
	    while count < 20:
	        y = model.decode(t)
	        y_list = list(y[0].data)
	        y_max = y_list.index(max(y_list))
	        
	        if id_w_dict[y_max] == '<EOS>':
	            break
	        
	        t = Variable(np.array([y_max], dtype='int32'))
	        #print(y_max, end=' ')
	        talk += id_w_dict[y_max]
	        
	        count += 1
	    #print()
	    return talk

