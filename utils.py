__author__ = 'jmh081701'
import  json

import  copy
import  numpy as np
import  random

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
        self.sentence_length = 100

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

        self.__load_test_data()

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
        with open(self.test_data_path,encoding='utf8') as fp:
            test_data_rawlines=fp.readlines()
        with open(self.test_label_path,encoding='utf8') as fp:
            test_label_rawlines=fp.readlines()
        total_lines = len(test_data_rawlines)
        assert len(test_data_rawlines)==len(test_label_rawlines)

        for index in range(total_lines):
            data_line = test_data_rawlines[index].split(" ")[:-1]
            label_line = test_label_rawlines[index].split(" ")[:-1]
            #assert len(data_line)==len(label_line)
            #align
            if len(data_line) < len(label_line):
                label_line=label_line[:len(data_line)]
            elif len(data_line)>len(label_line):
                data_line=data_line[:len(label_line)]
            assert len(data_line)==len(label_line)

            data=[int(self.word2id.get(each,0)) for each in data_line]
            label=[int(self.state.get(each,self.state['O'])) for each in label_line]
            self.test_data_raw.append(data)
            self.test_label_raw.append(label)


    def pad_sequence(self,sequence,object_length,pad_value=None):
        '''
        :param sequence: 待填充的序列
        :param object_length:  填充的目标长度
        :return:
        '''
        sequence =copy.deepcopy(sequence)
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
        efficient_sequence_length=[]
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
            efficient_sequence_length.append(min(100,len(labels[index])))
        return output_x,output_label,efficient_sequence_length
        #返回的都是下标,注意efficient_sequence_length是有效的长度

    def test_data(self):
        output_x=[]
        output_label=[]
        efficient_sequence_length=[]
        datas = self.test_data_raw[0:]
        labels = self.test_label_raw[0:]
        for index in range(len(datas)):
            #复制填充
            data= self.pad_sequence(datas[index],self.sentence_length)
            label = self.pad_sequence(labels[index],self.sentence_length)
            output_x.append(data)
            output_label.append(label)
            efficient_sequence_length.append(min(100,len(labels[index])))
        return output_x,output_label,efficient_sequence_length
    def next_valid_batch(self):
        output_x=[]
        output_label=[]
        efficient_sequence_length=[]
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
            efficient_sequence_length.append(min(100,len(labels[index])))
        return output_x,output_label,efficient_sequence_length


state={'O':0,'B-LOC':1,'I-LOC':2,'B-PER':3,'I-PER':4,'B-ORG':5,'I-ORG':6}
def extract_named_entity(labels,lens):
#输入是一个句子的标签
    B_PER=-1
    L_PER=-1

    B_LOC=-1
    L_LOC=-1

    B_ORG=-1
    L_ORG=-1
    rst = set()
    for index in range(lens):
        if labels[index]==state['O']:
            if B_PER >=0:
                rst.add(('PER',B_PER,L_PER))
                B_PER=-1
                L_PER=0
            if B_ORG >=0:
                rst.add(('ORG',B_ORG,L_ORG))
                B_ORG=-1
                L_ORG=0
            if B_LOC>=0:
                rst.add(('LOC',B_LOC,L_LOC))
                B_LOC=-1
                L_LOC=0
        if labels[index]==state['B-LOC']:
            if B_PER >=0:
                rst.add(('PER',B_PER,L_PER))
                B_PER=-1
                L_PER=0
            if B_ORG >=0:
                rst.add(('ORG',B_ORG,L_ORG))
                B_ORG=-1
                L_ORG=0
            if B_LOC>=0:
                rst.add(('LOC',B_LOC,L_LOC))
                B_LOC=-1
                L_LOC=0
            B_LOC=index
            L_LOC=1

        if labels[index]==state['B-PER']:
            if B_PER >=0:
                rst.add(('PER',B_PER,L_PER))
                B_PER=-1
                L_PER=0
            if B_ORG >=0:
                rst.add(('ORG',B_ORG,L_ORG))
                B_ORG=-1
                L_ORG=0
            if B_LOC>=0:
                rst.add(('LOC',B_LOC,L_LOC))
                B_LOC=-1
                L_LOC=0
            B_PER=index
            L_PER=1

        if labels[index]==state['B-ORG']:
            if B_PER >=0:
                rst.add(('PER',B_PER,L_PER))
                B_PER=-1
                L_PER=0
            if B_ORG >=0:
                rst.add(('ORG',B_ORG,L_ORG))
                B_ORG=-1
                L_ORG=0
            if B_LOC>=0:
                rst.add(('LOC',B_LOC,L_LOC))
                B_LOC=-1
                L_LOC=0
            B_ORG=index
            L_ORG=1

        if labels[index]==state['I-LOC']:
            if B_LOC>=0:
                L_LOC+=1
        if labels[index]==state['I-ORG']:
            if B_ORG>=0:
                L_ORG+=1

        if labels[index]==state['I-PER']:
            if B_PER>=0:
                L_PER+=1
    return  rst

def evaluate(predict_labels,real_labels,efficient_length):
#输入的单位是batch;
# predict_labels:[batch_size,sequence_length],real_labels:[batch_size,sequence_length]
    sentence_nums =len(predict_labels) #句子的个数
    predict_cnt=0
    predict_right_cnt=0
    real_cnt=0
    for sentence_index in range(sentence_nums):
        try:
            predict_set=extract_named_entity(predict_labels[sentence_index],efficient_length[sentence_index])
            real_set=extract_named_entity(real_labels[sentence_index],efficient_length[sentence_index])
            right_=predict_set.intersection(real_set)
            predict_right_cnt+=len(right_)
            predict_cnt += len(predict_set)
            real_cnt +=len(real_set)
        except Exception as exp:
            print(predict_labels[sentence_index])
            print(real_labels[sentence_index])
    precision = predict_right_cnt/(predict_cnt+0.000000000001)
    recall = predict_right_cnt/(real_cnt+0.000000000001)
    F1 = 2 * precision*recall/(precision+recall+0.00000000001)
    return {'precision':precision,'recall':recall,'F1':F1}

if __name__ == '__main__':
    dataGen = DATAPROCESS(train_data_path="data/source_data.txt",
                          train_label_path="data/source_label.txt",
                          test_data_path="data/test_data.txt",
                          test_label_path="data/test_label.txt",
                          word_embedings_path="data/source_data.txt.ebd.npy",
                          vocb_path="data/source_data.txt.vab",
                          batch_size=90,
                          seperate_rate=0.3
                        )
    datas,labels,efficient_sequence_length = dataGen.test_data()
    print(evaluate(labels,labels,efficient_sequence_length))

