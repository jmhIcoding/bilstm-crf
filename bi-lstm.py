#coding:utf-8
__author__ = 'jmh081701'
import  tensorflow as tf
from  baseTool import DATA_PREPROCESS
from tensorflow.contrib import  crf
import  numpy as np

def viterbi_decode(score, transition_params,supervised_y=None):
      """Decode the highest scoring sequence of tags outside of TensorFlow.
        修改crf里面的源码,本函数支持一个batch一个batch的解码
      """
      global sequence_length
      global O_index
      global dataGenerator
      shape = np.shape(score)
      viterbis=[]
      right_rate =0
      for i in range(shape[0]):
          #print(transition_params)
          viterbi,_=crf.viterbi_decode(score[i],transition_params)
          #print(score[i])
          viterbis.append(viterbi)

      if supervised_y is not None:
          #测试命名实体的准确率:看有多少个实体名称被识别出来
          namely_set_supervise=set()
          namely_set_predict=set()
          B_LOC=0
          B_PER=0
          B_ORG=0

          LOC_Len=0
          PER_Len=0
          ORG_Len=0
          for simple_index in range(np.shape(supervised_y)[0]):
                for col_index in range(np.shape(supervised_y)[1]):
                    if supervised_y[simple_index,col_index]==dataGenerator.state['B-LOC']:
                        B_LOC=col_index
                        LOC_Len=1
                    if supervised_y[simple_index,col_index]==dataGenerator.state['I-LOC']:
                        LOC_Len+=1

                    if supervised_y[simple_index,col_index]==dataGenerator.state['B-PER']:
                        B_PER=col_index
                        PER_Len=1
                    if supervised_y[simple_index,col_index]==dataGenerator.state['I-PER']:
                        PER_Len+=1

                    if supervised_y[simple_index,col_index]==dataGenerator.state['B-ORG']:
                        B_ORG=col_index
                        ORG_Len=1
                    if supervised_y[simple_index,col_index]==dataGenerator.state['I-ORG']:
                        ORG_Len+=1

                    if supervised_y[simple_index,col_index]==dataGenerator.state['O']:
                        if PER_Len==0 and LOC_Len==0 and ORG_Len==0:
                            continue
                        if PER_Len>0 and LOC_Len >0 and ORG_Len > 0:
                            #一定有错误
                            LOC_Len=0
                            PER_Len=0
                            continue
                        if PER_Len>0:
                            namely_set_supervise.add(('PER',simple_index,B_PER,PER_Len))
                        if LOC_Len>0:
                            namely_set_supervise.add(('LOC',simple_index,B_LOC,LOC_Len))
                        if ORG_Len>0:
                            namely_set_supervise.add(('ORG',simple_index,B_ORG,ORG_Len))
                        LOC_Len=0
                        PER_Len=0
                        ORG_Len=0
          B_LOC=0
          B_PER=0
          B_ORG=0

          LOC_Len=0
          PER_Len=0
          ORG_Len=0
          for simple_index in range(np.shape(supervised_y)[0]):
                for col_index in range(np.shape(supervised_y)[1]):
                    #print(viterbis[simple_index][0])
                    if viterbis[simple_index][col_index]==dataGenerator.state['B-LOC']:
                        B_LOC=col_index
                        LOC_Len=1
                    if viterbis[simple_index][col_index]==dataGenerator.state['I-LOC']:
                        LOC_Len+=1

                    if viterbis[simple_index][col_index]==dataGenerator.state['B-PER']:
                        B_PER=col_index
                        PER_Len=1
                    if viterbis[simple_index][col_index]==dataGenerator.state['I-PER']:
                        PER_Len+=1

                    if viterbis[simple_index][col_index]==dataGenerator.state['B-ORG']:
                        B_ORG=col_index
                        ORG_Len=1
                    if viterbis[simple_index][col_index]==dataGenerator.state['I-ORG']:
                        ORG_Len+=1

                    if viterbis[simple_index][col_index]==dataGenerator.state['O']:
                        if PER_Len>0 and LOC_Len >0 and ORG_Len>0:
                            #一定有错误
                            LOC_Len=0
                            PER_Len=0
                            ORG_Len=0
                            continue
                        if PER_Len>0:
                            namely_set_supervise.add(('PER',simple_index,B_PER,PER_Len))
                        if LOC_Len>0:
                            namely_set_supervise.add(('LOC',simple_index,B_LOC,LOC_Len))
                        if ORG_Len>0:
                            namely_set_predict.add(('ORG',simple_index,B_ORG,ORG_Len))
                        LOC_Len=0
                        PER_Len=0
                        ORG_Len=0
          SAME=namely_set_supervise.intersection(namely_set_predict)
          print('predict',namely_set_predict)
          print('surpervise',namely_set_supervise)
          PRECISION=len(SAME)/(0.000001+len(namely_set_predict))
          RECALL=len(SAME)/(0.000001+len(namely_set_supervise))
          F1_SCORE=2*PRECISION*RECALL/(PRECISION+RECALL+0.000001)
          right_rate =F1_SCORE
      return viterbis,right_rate

dataGenerator = DATA_PREPROCESS\
                (
                         train_data="data/source_data.txt",train_label="data/source_label.txt",
                         test_data="data/test_data.txt",test_label="data/test_label.txt",
                         embedded_words="data/source_data.txt.ebd.npy",
                         vocb="data/source_data.txt.vab"
                    )
O_index = dataGenerator.state['O']
train_rate=0.01
train_step=10000
batch_size=100
display_step=100

#每个词的词向量的长度
frame_size=dataGenerator.embedding_vec_length
#每个序列的长度,每句话的长度不定
sequence_length=dataGenerator.sequence_length

#前向和后向的LSTM 都是一层的
hidden_num=30
num_tags=dataGenerator.state_nums

#定义输入,输出,注意序列的长度是变化的。
input_x=tf.placeholder(dtype=tf.float32,shape=[None],name="input_x")
input_y=tf.placeholder(dtype=tf.int32,shape=[None,None],name="labels")
seq_lengths = tf.placeholder(dtype=tf.int32,shape=[None],name="batch_sequence_lengths") #专门提供给crf使用的
#定义P,A矩阵;
# P矩阵形状: 词的个数 X 状态数目:这个矩阵是计算出来的结果,不是以单独的矩阵出现的
# A矩阵形状: 状态数目 X 状态数目
#A=tf.Variable(tf.truncated_normal(stddev=0.1,shape=[num_tags,num_tags]))
#参数矩阵
Wc=tf.Variable(tf.truncated_normal(stddev=0.0056,shape=[2*num_tags,num_tags]))
bc=tf.Variable(tf.zeros(shape=[num_tags]))

V1=tf.Variable(tf.truncated_normal(stddev=0.0056,shape=[hidden_num,num_tags]))
V2=tf.Variable(tf.truncated_normal(stddev=0.0056,shape=[hidden_num,num_tags]))


#生成bi-lstm网络
with tf.name_scope("lstm"):
        x=tf.reshape(input_x,shape=[batch_size,sequence_length,frame_size])
        rnn_cell_fw=tf.nn.rnn_cell.LSTMCell(hidden_num)
        #前向RNN
        rnn_cell_bw=tf.nn.rnn_cell.LSTMCell(hidden_num)
        #后向RNN
        output,states=tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,rnn_cell_bw,x,dtype=tf.float32)
        #注意output有两部分：output_fw和output_bw.
        fw_output = output[0] #output[0]的形状：[batch_size, max_time, cell_fw.output_size(hidden_num)].
        bw_output = output[1] #与fw_output同理
        #各项拼接
        fw_output = tf.reshape(fw_output,shape=[-1,hidden_num],name="fw_output")
        bw_output = tf.reshape(bw_output,shape=[-1,hidden_num],name="bw_output")
        Lai=tf.matmul(fw_output,V1)
        Rai=tf.matmul(bw_output,V2)
        concat=tf.concat([Lai,Rai],-1)#[batch_size,sequence_length,2*num_tags]
        concat = tf.reshape(concat,[-1,2*num_tags])#[batch_size * sequence_length,2*num_tags]
        concat = tf.nn.dropout(concat,0.1)
        output=tf.tanh(tf.matmul(concat,Wc))                #+bc,'concat_op' #[batch_size * sequence_length,num_tags]
        P = tf.reshape(output,shape=[-1,sequence_length,num_tags],name="after_reshape") #恢复形状
#crf的log似然损失函数
#print crf_log_likelihood的参数
pred_p =P
print("#"*40)
print(pred_p)
print(seq_lengths)

log_likelihood,A=crf.crf_log_likelihood(inputs=pred_p,tag_indices=input_y,sequence_lengths=seq_lengths)
print(A)
cost = tf.reduce_mean(-log_likelihood)
train = tf.train.AdamOptimizer(train_rate).minimize(cost)

sess=tf.Session()
sess.run(tf.initialize_all_variables())
step=1
while step<train_step:
    batch_x,batch_y,batch_seq_lengths,____=dataGenerator.next_train_batch(batch_size)
#   batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
    _loss,__=sess.run([cost,train],feed_dict={input_x:batch_x,input_y:batch_y,seq_lengths:batch_seq_lengths})
    print(step,_loss)
    if step % display_step ==0:
        #计算一波正确率
        valid_x ,valid_y,batch_seq_lengths,___ = dataGenerator.next_train_batch(batch_size)
        scores,transition_parameter = sess.run([pred_p,A],feed_dict={input_x:valid_x,input_y:valid_y,seq_lengths:batch_seq_lengths})
        viterbi,right_rate = viterbi_decode(scores,transition_parameter,supervised_y=valid_y)
        print({"step":step,"right_rate":right_rate})
    step+=1
