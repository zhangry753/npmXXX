'''
Created on 2017年11月17日

@author: zry
'''
import numpy as np
import tensorflow as tf
import tflearn
import sonnet as snt
import config

class DQN:
  def __init__(self,
               register,reg_usage,target,actions,rewards,terminals,
               discount=0.9,
               clip_delta=0,
               sum_along_batch=False,
               sum_along_time=False,
               global_step=None,
               lr=0.03, rms_decay=0.9, rms_eps=1e-10,
               name="DQN"):
    self.network_name = name
    num_act = (config.instruction_size-1)*config.parameter_size
#     register = tf.placeholder('float32', [None, config.batch_size, config.register_num*config.register_size], name=self.network_name + '_reg')
#     reg_usage = tf.placeholder('float32', [None, config.batch_size, config.register_num], name=self.network_name + '_regUsage')
#     target = tf.placeholder('float32', [None, config.batch_size, config.num_bits], name=self.network_name + 'target')
#     actions = tf.placeholder("float32", [None, num_act], name=self.network_name + '_actions')
#     rewards = tf.placeholder("float32", [None, config.batch_size], name=self.network_name + '_rewards')
#     terminals = tf.placeholder("float32", [None, 1], name=self.network_name + '_terminals')


#     # tensor layer
#     tensor_layer_size = config.register_num
#     register_flat = tf.reshape(register, [-1,config.register_num*config.register_size])
#     target_flat = tf.reshape(target, [-1,config.num_bits])
#     hidden_concat = tflearn.fully_connected(tf.concat([register_flat,target_flat],-1), tensor_layer_size)
#     hidden_tensor = tf.nn.relu(tf.multiply(
#         tflearn.fully_connected(hidden_concat, tensor_layer_size),
#         hidden_concat
#         ))
#     hidden_tensor = tf.reshape(hidden_tensor, [-1,config.batch_size,tensor_layer_size])
#     # lstm
#     lstm_layer_size = num_act*2
#     hidden_lstm_bf,_ = tf.nn.bidirectional_dynamic_rnn(
#         tflearn.BasicLSTMCell(lstm_layer_size), 
#         tflearn.BasicLSTMCell(lstm_layer_size),
#         tf.concat([hidden_tensor,reg_usage],-1),
#         time_major=True,
#         dtype=tf.float32)
#     hidden_lstm = tf.concat(hidden_lstm_bf, -1)
#     hidden_lstm_flat = tf.reshape(hidden_lstm, [-1,lstm_layer_size*2])
#     # FC1
#     FC1_layer_size = num_act*2
#     hidden_fc1 = tflearn.fully_connected(hidden_lstm_flat, FC1_layer_size, activation=tf.nn.relu)
#     # FC2
#     FC2_layer_size = num_act
#     y_flat = tflearn.fully_connected(hidden_fc1, FC2_layer_size)
#     self.y = tf.reshape(y_flat, [-1,config.batch_size,FC2_layer_size])
    
    
    register_flat = tf.reshape(register, [-1,config.register_num*config.register_size])
    target_flat = tf.reshape(target, [-1,config.num_bits])
    reg_usage_flat = tf.reshape(reg_usage, [-1,config.register_num])
    _input = tf.concat([register_flat, target_flat, reg_usage_flat], -1)
    hidden_fc1 = tflearn.fully_connected(_input, 256, activation=tf.nn.relu)
    hidden_fc2 = tflearn.fully_connected(hidden_fc1, 512, activation=tf.nn.relu)
    hidden_fc3 = tflearn.fully_connected(hidden_fc2, 128, activation=tf.nn.relu)
    hidden_fc4 = tflearn.fully_connected(_input, num_act)
    self.y = tf.reshape(hidden_fc4, [-1,config.batch_size,num_act])
    self.y = tf.Variable(tf.zeros([config.instruction_length,config.batch_size,num_act]))
    
    y_ins_para = tf.reduce_mean(tf.reshape(self.y, [-1, config.batch_size, config.instruction_size-1, config.parameter_size]), 1)
    self.ins_pred = tf.argmax(tf.reduce_sum(y_ins_para, -1),-1)
    self.para_pred = tf.argmax(tf.reduce_sum(y_ins_para, -2),-1)
#     # select action
#     if np.random.rand() > self.params['eps']:
#       #greedy with random tie-breaking
#       Q_pred = self.sess.run(self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0] 
#       a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
#       if len(a_winner) > 1:
#         act_idx = a_winner[np.random.randint(0, len(a_winner))][0]
#         return act_idx,self.engine.legal_actions[act_idx], np.amax(Q_pred)
#       else:
#         act_idx = a_winner[0][0]
#         return act_idx,self.engine.legal_actions[act_idx], np.amax(Q_pred)
#     else:
#       #random
#       act_idx = np.random.randint(0,len(self.engine.legal_actions))
#       Q_pred = self.sess.run(self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0]
#       return act_idx,self.engine.legal_actions[act_idx], Q_pred[act_idx]
    # Q,Cost,Optimizer
    self.q_t = tf.concat([tf.reduce_max(self.y, -1)[1:,:] ,tf.zeros([1,config.batch_size])], 0)
    self.yj = rewards + (1.0-terminals)*discount*self.q_t
    self.Qxa = self.y * tf.expand_dims(actions, 1)
    self.Q_pred = tf.reduce_max(self.Qxa, reduction_indices=-1)
    diff = self.yj - self.Q_pred
    # diff_square clip
    if clip_delta > 0 :
      quadratic_part = tf.minimum(tf.abs(diff), tf.constant(clip_delta))
      linear_part = tf.subtract(tf.abs(diff),quadratic_part)
      diff_square = 0.5 * tf.pow(quadratic_part,2) + clip_delta*linear_part
    else:
      diff_square = tf.multiply(tf.constant(0.5),tf.pow(diff, 2))
    # merge along timestep and batch 
    if sum_along_time:
      cost_time_merge = tf.reduce_sum(diff_square, 0)
    else:
      cost_time_merge = tf.reduce_mean(diff_square, 0)
    if sum_along_batch:
      self.cost = tf.reduce_sum(cost_time_merge)
    else:
      self.cost = tf.reduce_mean(cost_time_merge)
    # optimizer
    self.train_step = tf.train.RMSPropOptimizer(lr,rms_decay,0.0,rms_eps).minimize(self.cost,global_step=global_step)
#     self.train_step = tf.train.AdamOptimizer().minimize(self.cost,global_step=global_step)

    
  
if __name__=="__main__":
  DQN()
    
