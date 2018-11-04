__author__ = 'jmh081701'
import  numpy as np
import  random
import json
class DATA_PREPROCESS:
    def __init__(self,train_data,train_label,test_data,test_label,embedded_words,vocb,seperate_rate=0.1,state={'O','B-LOC','I-LOC','B-PER','I-PER'}):
        self.train_data_file = train_data
        self.train_label_file = train_label
        self.test_data_file = test_data
        self.test_label_file = test_label
        self.word2vec_file = embedded_words
        self.vocb_file = vocb

        # 载入词向量
        self.word2vec=np.load(self.word2vec_file)

        with open(self.vocb_file,encoding='utf8') as fp:
            self.index2word = json.load(fp)

        self.words={}
        for key in self.index2word:
            word=self.index2word[key]
            self.words.setdefault(word,key)

        #处理隐状态
        self.state={}
        if state != None:
            for each in state:
                self.state.setdefault(each,len(self.state))
        #载入训练集
        with open(self.train_data_file,encoding='utf8') as fp:
            train_raw_data = fp.readlines()
            train_lines =[]
            for line in train_raw_data:
                raw_words = line.split(" ")
                words = [self.word2index(word) for word in raw_words ]
                train_lines.append(words)
            self.train_data = train_lines

        with open(self.train_label_file,encoding='utf8') as fp:
            train_raw_label = fp.readlines()
            train_labels =[]
            for line in train_raw_label:
                raw_labels = line.split(" ")
                if self.state != None:
                    labels = [self.state.get(label,self.state['O']) for label in raw_labels]
                else:
                    labels=[]
                    for label in raw_labels:
                        if label in self.state:
                            labels.append( self.state[label])
                        else:
                            self.state.setdefault(label,len(self.state))
                            labels.append(self.state[label])
                train_labels.append(labels)
            self.train_labels =train_labels

        #划分训练集为 训练集和验证集
            self.train_set=set()
            self.valid_set=set()

        while len( self.valid_set ) < int(seperate_rate * len(self.train_data)):
            index = random.randint(0,len(self.train_data)-1)
            self.valid_set.add(index)

        #载入测试集
        with open(self.test_data_file,encoding='utf8') as fp:
            test_raw_data = fp.readlines()
            test_lines =[]
            for line in test_raw_data:
                raw_words = line.split(" ")
                words = [self.word2index(word) for word in raw_words ]
                train_lines.append(words)
            self.test_lines = test_lines

        with open(self.test_label_file,encoding='utf8') as fp:
            test_raw_label = fp.readlines()
            test_labels =[]
            for line in test_raw_label:
                raw_labels = line.split(" ")
                labels = [self.state.get(label,self.state['O']) for label in raw_labels]
                test_lines.append(labels)
            self.test_labels =test_labels

    def word2index(self,word):
        return self.words.get(word,self.words['<UNK>'])
    def lookup(self,word):
        return self.word2vec(self.word2index(word))

    def next_train_batch(self,batch_size):
        x=[]
        y=[]
        while len(x) < batch_size:
            index = random.randint(0,len(self.train_data)-1)
            if not index in self.valid_set:
                _label = self.train_labels[index]
                _x=self.train_data[index]
                for i in range(len(_x)):
                    _x[i]=self.word2vec[ int(_x[i]) ]
                x.append(_x)
                y.append(_label)
        return x,y

    def next_valid_batch(self,batch_size):
        x=[]
        y=[]
        while len(x) < batch_size:
            index = random.randint(0,len(self.valid_set)-1)
            if index in self.valid_set:
                _label = self.train_labels[index]
                _x=self.train_data[index]
                for i in range(len(_x)):
                    _x[i]=self.word2vec[int( _x[i] )]
                x.append(_x)
                y.append(_label)
        return x,y

    def test(self):
        x=[]
        y=[]
        for index in range(len(self.test_lines)):
                _label = self.test_labels[index]
                _x=self.test_lines[index]
                for i in range(len(_x)):
                    _x[i]=self.word2vec[ int(_x[i] )]
                x.append(_x)
                y.append(_label)
        return x,y
if __name__ == '__main__':
    data=DATA_PREPROCESS(train_data="data/source_data.txt",train_label="data/source_label.txt",
                         test_data="data/tes_datat.txt",test_label="data/test_label.txt",
                         embedded_words="data/source_data.txt.ebd.npy",
                         vocb="data/source_data.txt.vab")
    x,y=data.next_valid_batch(batch_size=2)
    print(x)
    print(y)

