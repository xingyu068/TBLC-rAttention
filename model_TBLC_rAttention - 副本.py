# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:50:09 2019
@author: xingyu

"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class TBLC_rAttention_config(object):  

    embedding_size = 100 
    seq_length = 50    
    num_classes = 3                       
    vocab_size = 5000
    num_layers= 1              
    rnn_num_hidden = embedding_size   
    num_filters = rnn_num_hidden*2 
    filter_size = 5              
    dropout_keep_prob = 0.8 
    learning_rate = 1e-3    
    batch_size = 100         
    print_per_batch = 100    
    save_per_batch = 10          
    beta_l2 = 1 
    n = seq_length 
    d_a = embedding_size  
    u = d_a 
    r = 3   
  
        
class TBLC_rAttention(object):  

    def __init__(self, config):  
        self.config = config  
        self.global_step = tf.Variable(0, trainable=False)               
        self.tblc_rAttention()

    def tblc_rAttention(self):

        initializer=tf.contrib.layers.xavier_initializer()
          
        with tf.name_scope('placeholders'):
            self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name="input_x")
            self.input_y = tf.placeholder(tf.int32, [None,self.config.num_classes], name="input_y")
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  
            self.x_len = tf.reduce_sum(tf.sign(self.input_x), 1, name='x_len')  
  
           
        with tf.name_scope("embedding_layer"):
            init_embeddings = tf.random_uniform([self.config.vocab_size, self.config.embedding_size])
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.embedding_words = tf.nn.embedding_lookup(self.embeddings, self.input_x)   
            print('===============self.embedding_words===========')
            print(self.embedding_words) 
                   
                
        with tf.name_scope("bilstm_layer"): 
            fw_cells_rnn = [rnn.LSTMCell(self.config.rnn_num_hidden) for _ in range(self.config.num_layers)] 
            bw_cells_rnn = [rnn.LSTMCell(self.config.rnn_num_hidden) for _ in range(self.config.num_layers)]
            fw_cells = tf.contrib.rnn.MultiRNNCell(fw_cells_rnn) 
            bw_cells = tf.contrib.rnn.MultiRNNCell(bw_cells_rnn)

            self.bilstm_outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_dynamic_rnn([fw_cells],[bw_cells], self.embedding_words, dtype=tf.float32)           
            print('===============self.bilstm_outputs.shap===========')
            print(self.bilstm_outputs.shape)           


        
        with tf.variable_scope("attention_layer"):   
           
            W_s1 = tf.get_variable('W_s1', shape=[self.config.d_a, 2*self.config.u],initializer=initializer)
            W_s2 = tf.get_variable('W_s2', shape=[self.config.r*self.config.seq_length, self.config.d_a],initializer=initializer) 
            A = tf.nn.softmax(tf.matmul(W_s2,tf.tanh(tf.matmul(W_s1, tf.reshape(self.bilstm_outputs, [2*self.config.u, -1])))))
            A = tf.reshape(A, shape=[-1, self.config.r*self.config.seq_length, self.config.n], name='attention_xi')
            self.attention_out = tf.matmul(A, self.bilstm_outputs, name='attention_out')   
            print('================attention_out.shap===========')
            print(self.attention_out.shape)  
      

        with tf.name_scope("cnn_layer"):
              
            conv = tf.layers.conv1d(self.attention_out, self.config.num_filters, self.config.filter_size, name='convolution')
            self.cnn_out = tf.reduce_max(conv, reduction_indices=[1], name='pooling')
            print('================conv.shap===========')
            print(conv.shape)         
            print('================cnn_out.shap===========')
            print(self.cnn_out.shape) 


        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.cnn_out, self.keep_prob)



        with tf.name_scope("output"):        
            self.logits = tf.layers.dense(self.h_drop , self.config.num_classes,name='y_output')           
            self.probability = tf.nn.softmax(self.logits, name="probability")  
            self.y_pred_cls = tf.argmax(self.probability, 1, name="predictions")      



        with tf.name_scope("optimize"):

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y, name="cross_entropy")
            self.loss = tf.reduce_mean(cross_entropy, name="loss")
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss, global_step=self.global_step)
  
          
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls, name="accuracy_in")
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy_out")
