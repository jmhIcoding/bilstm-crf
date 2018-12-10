__author__ = 'jmh081701'
import  tensorflow as tf
from  tensorflow.contrib import  crf
import  random
from  utils import *
import logging
import datetime

logging.basicConfig(level=logging.INFO,format="%(asctime)s  - %(message)s")
logger = logging.getLogger(__name__)

#超参数
batch_size=100
dataGen = DATAPROCESS(train_data_path="data/source_data.txt",
                          train_label_path="data/source_label.txt",
                          test_data_path="data/test_data.txt",
                          test_label_path="data/test_label.txt",
                          word_embedings_path="data/source_data.txt.ebd.npy",
                          vocb_path="data/source_data.txt.vab",
                          batch_size=batch_size
                        )
#模型超参数
tag_nums =len(dataGen.state)    #标签数目
hidden_nums = 650                #bi-lstm的隐藏层单元数目
learning_rate = 0.00075          #学习速率
sentence_len = dataGen.sentence_length #句子长度,输入到网络的序列长度
frame_size = dataGen.embedding_length #句子里面每个词的词向量长度

#网络的变量
word_embeddings =  tf.Variable(initial_value=dataGen.word_embeddings,trainable=True) #参与训练
#输入占位符
input_x = tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_word_id')#输入词的id
input_y = tf.placeholder(dtype=tf.int32,shape=[None,sentence_len],name='input_labels')
sequence_lengths=tf.placeholder(dtype=tf.int32,shape=[None],name='sequence_lengths_vector')
#
with tf.name_scope('projection'):
    #投影层,先将输入的词投影成相应的词向量
    word_id = input_x
    word_vectors = tf.nn.embedding_lookup(word_embeddings,ids=word_id,name='word_vectors')
    word_vectors = tf.nn.dropout(word_vectors,0.8)
with tf.name_scope('bi-lstm'):

    labels = tf.reshape(input_y,shape=[-1,sentence_len],name='labels')
    fw_lstm_cell =tf.nn.rnn_cell.LSTMCell(hidden_nums)
    bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_nums)
    #双向传播
    output,_state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,bw_lstm_cell,inputs=word_vectors,sequence_length=sequence_lengths,dtype=tf.float32)
    fw_output = output[0]#[batch_size,sentence_len,hidden_nums]
    bw_output =output[1]#[batch_size,sentence_len,hidden_nums]
    V1=tf.get_variable('V1',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[hidden_nums,hidden_nums])
    V2=tf.get_variable('V2',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[hidden_nums,hidden_nums])
    fw_output = tf.reshape(tf.matmul(tf.reshape(fw_output,[-1,hidden_nums],name='Lai') , V1),shape=tf.shape(output[0]))
    bw_output = tf.reshape(tf.matmul( tf.reshape(bw_output,[-1,hidden_nums],name='Rai') , V2),shape=tf.shape(output[1]))
    contact = tf.concat([fw_output,bw_output],-1,name='bi_lstm_concat')#[batch_size,sentence_len,2*hidden_nums]
    contact = tf.nn.dropout(contact,0.9)
    s=tf.shape(contact)
    contact_reshape=tf.reshape(contact,shape=[-1,2*hidden_nums],name='contact')
    W=tf.get_variable('W',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[2*hidden_nums,tag_nums],trainable=True)
    b=tf.get_variable('b',initializer=tf.zeros(shape=[tag_nums]))
    p=tf.nn.relu(tf.matmul(contact_reshape,W)+b)
    logit= tf.reshape(p,shape=[-1,s[1],tag_nums],name='omit_matrix')


with tf.name_scope("crf") :
    log_likelihood,transition_matrix=crf.crf_log_likelihood(logit,labels,sequence_lengths=sequence_lengths)
    cost = -tf.reduce_mean(log_likelihood)
with tf.name_scope("train-op"):
    global_step = tf.Variable(0,name='global_step',trainable=False)
    optim = tf.train.AdamOptimizer(learning_rate)
    #train_op=optim.minimize(cost)
    grads_and_vars = optim.compute_gradients(cost)
    grads_and_vars = [[tf.clip_by_value(g,-5,5),v] for g,v in grads_and_vars]
    train_op = optim.apply_gradients(grads_and_vars,global_step)
#
#载入模型如果有参数的话
checkpoint_prefix="paras/bilstm-crf-models"
saver = tf.train.Saver()

display_step = len(dataGen.train_batches)
epoch_nums = 60 #迭代的数据轮数
max_batch = len(dataGen.train_batches)*epoch_nums
step=1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cpkt=tf.train.get_checkpoint_state(checkpoint_prefix)
    if  cpkt and cpkt.model_checkpoint_path:
        saver.restore(sess,cpkt.model_checkpoint_path)
        logging.info("restore from history models.")
    else:
        logging.warning("retrain a models.")
    while step<max_batch:
        batch_x,batch_y,efficient_sequence_length = dataGen.next_train_batch()
        _,loss,score=sess.run([train_op,cost,logit],{input_x:batch_x,input_y:batch_y,sequence_lengths:efficient_sequence_length})
        logging.info({'loss':loss,'step':step})
        if(step % display_step ==0):
            valid_x,valid_y,efficient_sequence_length=dataGen.next_valid_batch()
            scores,transition_matrix_out=sess.run([logit,transition_matrix],{input_x:valid_x,input_y:valid_y,sequence_lengths:efficient_sequence_length})
            for i in range(batch_size):
                label,_=crf.viterbi_decode(scores[i],transition_params=transition_matrix_out)
                label=label[:efficient_sequence_length[i]]
                print(label)
            logger.info("Save a stage model para for %d epoch."%(step/display_step))
            saver.save(sess,checkpoint_prefix)
        step+=1
    saver.save(sess,checkpoint_prefix)
    logger.info("save models well.")
    data_x,label_y,efficient_sequence_length=dataGen.test_data()
    scores,transition_matrix_out=sess.run([logit,transition_matrix],{input_x:data_x,input_y:label_y,sequence_lengths:efficient_sequence_length})
    real_labels = label_y
    predict_labels =[]
    for i in range(len(scores)):
        labels,_=crf.viterbi_decode(scores[i],transition_matrix_out)
        predict_labels.append(labels)
    print("====================TEST======================")
    print(evaluate(predict_labels,real_labels,efficient_sequence_length))
    print("===================END MODEL==================")

