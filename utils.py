__author__ = 'jmh081701'
import  json
import os
import  sys
import  numpy as np
import  random
import tensorflow as tf
import  logging
class  DATAPROCESS:
    def __init__(self,train_data_path,train_label_path,test_data_path,test_label_path,word_embedings_path,vocb_path,seperate_rate=0.1,batch_size=100,
                 state={'O':0,'B-LOC':1,'I-LOC':2,'B-PER':3,'I-PER':4,'B-ORG':5,'I-ORG':6}):
        self.train_data_path =train_data_path
        self.train_label_path =train_label_path
        self.test_data_path = test_data_path
        self.test_label_path = test_label_path
        self.word_embedding_path = word_embedings_path
        self.vocb_path  = vocb_path
        self.state = state
        self.seperate_rate =seperate_rate
        self.batch_size = batch_size
        self.sentence_length = 20

        #data structure to build
        self.train_data_raw=[]
        self.train_label_raw =[]
        self.valid_data_raw=[]
        self.valid_label_raw = []

        self.test_data_raw =[]
        self.test_label_raw =[]

        self.word_embeddings=None
        self.id2word=None
        self.word2id=None
        self.embedding_length =0

        self.__load_wordebedding()


        self.__load_train_data()

        #self.__load_test_data()

        self.last_batch=0
    def __load_wordebedding(self):
        self.word_embeddings=np.load(self.word_embedding_path)
        self.embedding_length = np.shape(self.word_embeddings)[-1]
        with open(self.vocb_path,encoding="utf8") as fp:
            self.id2word = json.load(fp)
        self.word2id={}
        for each in self.id2word:
            self.word2id.setdefault(self.id2word[each],each)

    def __load_train_data(self):

        with open(self.train_data_path,encoding='utf8') as fp:
            train_data_rawlines=fp.readlines()
        with open(self.train_label_path,encoding='utf8') as fp:
            train_label_rawlines=fp.readlines()
        total_lines = len(train_data_rawlines)
        assert len(train_data_rawlines)==len(train_label_rawlines)

        for index in range(total_lines):
            data_line = train_data_rawlines[index].split(" ")[:-1]
            label_line = train_label_rawlines[index].split(" ")[:-1]
            #assert len(data_line)==len(label_line)
            #align
            if len(data_line) < len(label_line):
                label_line=label_line[:len(data_line)]
            elif len(data_line)>len(label_line):
                data_line=data_line[:len(label_line)]
            assert len(data_line)==len(label_line)
            #add and seperate valid ,train set.
            data=[int(self.word2id.get(each,0)) for each in data_line]
            label=[int(self.state.get(each,self.state['O'])) for each in label_line]
            if random.uniform(0,1) <self.seperate_rate:
                self.valid_data_raw.append(data)
                self.valid_label_raw.append(label)
            else:
                self.train_data_raw.append(data)
                self.train_label_raw.append(label)
        self.train_batches= [i for i in range(int(len(self.train_data_raw)/self.batch_size) -1)]
        self.train_batch_index =0
        self.valid_batches=[i for i in range(int(len(self.valid_data_raw)/self.batch_size) -1) ]
        self.valid_batch_index = 0
    def __load_test_data(self):
        pass

    def pad_sequence(self,sequence,object_length,pad_value=None):
        '''
        :param sequence: 待填充的序列
        :param object_length:  填充的目标长度
        :return:
        '''
        if pad_value is None:
            sequence = sequence*(1+int((0.5+object_length)/(len(sequence))))
            sequence = sequence[:object_length]
        else:
            sequence = sequence+[pad_value]*(object_length- len(sequence))
        return sequence

    def next_train_batch(self):
        #padding
        output_x=[]
        output_label=[]
        index =self.train_batches[self.train_batch_index]
        self.train_batch_index =(self.train_batch_index +1 ) % len(self.train_batches)
        datas = self.train_data_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.train_label_raw[index*self.batch_size:(index+1)*self.batch_size]
        for index in range(self.batch_size):
            #复制填充
            data= self.pad_sequence(datas[index],self.sentence_length)
            label = self.pad_sequence(labels[index],self.sentence_length)
            output_x.append(data)
            output_label.append(label)
        return output_x,output_label
        #返回的都是下标
    def next_test_batch(self):
        pass
    def next_valid_batch(self):
        output_x=[]
        output_label=[]
        index =self.valid_batches[self.valid_batch_index]
        self.valid_batch_index =(self.valid_batch_index +1 ) % len(self.valid_batches)
        datas = self.valid_data_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.valid_label_raw[index*self.batch_size:(index+1)*self.batch_size]
        for index in range(self.batch_size):
            #复制填充
            data= self.pad_sequence(datas[index],self.sentence_length)
            label = self.pad_sequence(labels[index],self.sentence_length)
            output_x.append(data)
            output_label.append(label)
        return output_x,output_label

if __name__ == '__main__':
    dataGen = DATAPROCESS(train_data_path="data/source_data.txt",
                          train_label_path="data/source_label.txt",
                          test_data_path="data/test_data.txt",
                          test_label_path="data/test_label.txt",
                          word_embedings_path="data/source_data.txt.ebd.npy",
                          vocb_path="data/source_data.txt.vab",
                          batch_size=15,
                          seperate_rate=0.3
                        )
    label2tag={}
    for each in dataGen.state:
        label2tag[dataGen.state[each]]=each
    with open("train_data","w",encoding='utf8') as fp:
        for i in range(len(dataGen.train_data_raw)):
            for j in range( len(dataGen.train_data_raw[i])):
                char = dataGen.id2word.get(str(dataGen.train_data_raw[i][j]),"<UNK>")
                label = label2tag[dataGen.train_label_raw[i][j]]
                fp.writelines("%s %s\n"%(char,label))
            fp.writelines("\n")
    with open("test_data","w",encoding='utf8') as fp:
        for i in range(len(dataGen.valid_data_raw)):
            for j in range(len(dataGen.valid_data_raw[i])):
                char = dataGen.id2word.get(str(dataGen.valid_data_raw[i][j]),"<UNK>")
                label = label2tag[dataGen.valid_label_raw[i][j]]
                fp.writelines("%s %s\n"%(char,label))
            fp.writelines("\n")
