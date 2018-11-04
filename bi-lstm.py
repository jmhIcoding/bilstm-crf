#coding:utf-8
__author__ = 'jmh081701'
import  tensorflow as tf
from  baseTool import DATA_PREPROCESS
from tensorflow.contrib import  crf
import  numpy as np


def viterbi_decode(score, transition_params):
      """Decode the highest scoring sequence of tags outside of TensorFlow.
        修改crf里面的源码,本函数支持一个batch一个batch的解码
      """
      shape = np.shape(score)
      viterbis=[]
      for i in range(shape[0]):
          trellis = np.zeros_like(score)
          backpointers = np.zeros_like(score, dtype=np.int32)
          trellis[0] = score[0]

          for t in range(1, score.shape[0]):
            v = np.expand_dims(trellis[t - 1], 1) + transition_params
            trellis[t] = score[t] + np.max(v, 0)
            backpointers[t] = np.argmax(v, 0)

          viterbi = [np.argmax(trellis[-1])]
          for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
          viterbi.reverse()
          viterbis.append(viterbi)
      return viterbis


def lstm(x,A,W):
    with tf.name_scope("lstm"):
        x=tf.reshape(x,shape=[-1,-1,frame_size])
        rnn_cell_fw=tf.nn.rnn_cell.LSTMCell(hidden_num)
        #前向RNN
        rnn_cell_bw=tf.nn.rnn_cell.LSTMCell(hidden_num)
        #后向RNN
        # 其实这是一个双向深度RNN网络,对于每一个长度为n的序列[x1,x2,x3,...,xn]的每一个xi,都会在深度方向跑一遍RNN,跑上hidden_num个隐层单元

        output,states=tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,rnn_cell_bw,x,dtype=tf.float32)
        #注意output有两部分：output_fw和output_bw.
        #states这个中间状态输出不管
        #将output[0]和Output[1]拼接在一起


        fw_output = output[0][:,:,-1] #output[0]的形状：[batch_size, max_time, cell_fw.output_size]
        # 所以 取各个batch,各个时间步里面的最后一个隐藏层的输出.
        bw_output = output[1][:,:,-1] #与fw_output同理

        #各项拼接
        output=tf.concat([fw_output,bw_output],1)#[batch_size,sequence_length,2]

        P= tf.tanh(tf.matmul(output,W),name="P")#[batch_size,sequence_length,num_tags]每个P[i]就是一个序列的P矩阵
        #这个P矩阵就是将来需要丢到crf里面的输入之一
        #得到一个输入sequence_length,形状是[batch_size]
        sequence_length=[]
        batch=tf.shape(x)[0]
        for i in range(batch):
            sequence_length.append(tf.shape(x[i])[0])

        return P,tf.to_float(sequence_length)


dataGenerator = DATA_PREPROCESS(train_data="data/source_data.txt",train_label="data/source_label.txt",
                         test_data="data/tes_datat.txt",test_label="data/test_label.txt",
                         embedded_words="data/source_data.txt.ebd.npy",
                         vocb="data/source_data.txt.vab"
                    )


train_rate=0.001
train_step=100
batch_size=2
display_step=10

#每个词的词向量的长度
frame_size=dataGenerator.embedding_vec_length
#每个序列的长度,每句话的长度不定
sequence_length=None

#前向和后向的LSTM都是一层的
hidden_num=1
num_tags=dataGenerator.state_nums

#定义输入,输出,注意序列的长度是变化的。
x=tf.placeholder(dtype=tf.float32,shape=[None,None,frame_size],name="inputx")
y=tf.placeholder(dtype=tf.float32,shape=[None,num_tags],name="expected_y")
#定义P,A矩阵;
# P矩阵形状: 词的个数 X 状态数目:这个矩阵是计算出来的结果,不是以单独的矩阵出现的
# A矩阵形状: 状态数目 X 状态数目
A=tf.Variable(tf.truncated_normal(stddev=0.1,shape=[num_tags,num_tags]))
#W矩阵,bi-LSTM的每个时间步乘以W
W=tf.Variable(tf.truncated_normal(stddev=0.1,shape=[2,num_tags]))

#生成bi-lstm网络
pred_p,seq_lengths=lstm(x,A,W)
#crf的log似然损失函数
cost=crf.crf_log_likelihood(inputs=pred_p,tag_indices=y,sequence_lengths=seq_lengths)
train=tf.train.AdamOptimizer(train_rate).minimize(cost)

sess=tf.Session()
sess.run(tf.initialize_all_variables())
step=1
while step<train_step:
    batch_x,batch_y=dataGenerator.next_train_batch(batch_size)
#    batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
    _loss,__=sess.run([cost,train],feed_dict={x:batch_x,y:batch_y})
    if step % display_step ==0:
        #计算一波正确率
        valid_x ,valid_y = dataGenerator.next_test_batch(batch_size)
        scores,transition_parameter = sess.run([pred_p,A],feed_dict={x:valid_x,y:valid_y})
    step+=1