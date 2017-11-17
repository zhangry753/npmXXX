'''
Created on 2017年10月31日

@author: zry
'''
import collections
import sonnet as snt
import tensorflow as tf

NPM_State = collections.namedtuple('AIPState', ('memory', 'register','register_usage','cur_ins_addr','cur_mem_addr'))

def encode(input_, target_shape):
    input_ = tf.cast(input_, tf.float32)
    zero_shape = input_.get_shape().as_list()
    if zero_shape[-1] < target_shape:
      zero_shape[-1] = target_shape - zero_shape[-1]
      zeros = tf.zeros(zero_shape)
      return  tf.concat([input_, zeros], -1)
    else:
      zero_shape[-1] = target_shape
      return tf.slice(input_, [0]*len(zero_shape), zero_shape)
def onehot_max(input_, min_value=0.8):
  input_softmax = tf.nn.softmax(input_)
  condition = tf.reduce_max(input_, -1) >= min_value
  input_onehot = tf.one_hot(tf.argmax(input_softmax, -1), input_.get_shape()[-1].value)
  return tf.where(condition, input_onehot, input_softmax)
def relu1(input_):
  return tf.where(input_>=1, tf.ones(tf.shape(input_)), tf.zeros(tf.shape(input_)))

class Neural_programming_machine(snt.RNNCore):
  
  def __init__(self,
               output_size, 
               memory_size, 
               memory_length, 
               register_size, 
               register_num,
               instruction_size, 
               instruction_length, 
               parameter_size, 
               parameter_num,
               name="AI_programming"):
    super(Neural_programming_machine, self).__init__(name=name)
    self._output_size = output_size
    self.mem_size = memory_size
    self.mem_len = memory_length
    self.reg_size = register_size
    self.reg_num = register_num
    self.ins_size = instruction_size
    self.ins_len = instruction_length
    self.para_size = parameter_size
    self.para_num = parameter_num
    self.instructions = tf.get_variable("instructions", [self.ins_len, self.ins_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))
    self.parameters = tf.get_variable("parameters", [self.ins_len, self.para_num, self.para_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))
        
    
  def _build(self, inputs, prev_state):
    memory=prev_state.memory
    register=prev_state.register
    reg_usage = prev_state.register_usage
    cur_ins_addr=prev_state.cur_ins_addr
    cur_mem_addr=prev_state.cur_mem_addr
    
    ins_clip = tf.clip_by_value(self.instructions, -10.0, 10.0)
    ins_softmax = tf.nn.softmax(ins_clip)
    para_clip = tf.clip_by_value(self.parameters, -10.0, 10.0)
    cur_ins = tf.matmul(cur_ins_addr, ins_softmax)
    parameters_flat = tf.reshape(para_clip, [self.ins_len, self.para_num*self.para_size])
    cur_para = tf.reshape(tf.matmul(cur_ins_addr, parameters_flat), [-1,self.para_num,self.para_size])
    ins = [tf.expand_dims(item,1) for item in tf.unstack(cur_ins, axis=1)]
    para = tf.unstack(cur_para, axis=1)
    #----------------------------------更新寄存器---------------------------------------------
    # ins0 输入到寄存器: para0--寄存器id
    input_value = encode(inputs, self.reg_size)
    new_reg_value = ins[0] * input_value
    reg_id = tf.nn.softmax(encode(para[0], self.reg_num))
#     reg_id_usable = reg_id * (1-reg_usage)
    reg_usage = reg_id + (1-reg_id)*reg_usage
    new_reg_value_all = tf.matmul(tf.expand_dims(reg_id,2), tf.expand_dims(new_reg_value,1)) +\
            tf.expand_dims(1-reg_id,2) * register
    register = tf.expand_dims(ins[0],1) * new_reg_value_all  +\
            tf.expand_dims(1-ins[0],1) * register
    # ins1 输出: para0--寄存器id
    reg_id = tf.nn.softmax(encode(para[0], self.reg_num))
#     reg_id_usable = reg_id * reg_usage
    reg_value = tf.reshape(tf.matmul(tf.expand_dims(reg_id,1), register), [-1,self.reg_size])
    output = ins[1] * encode(reg_value, self._output_size)
    # ins2 终止(重复执行本条指令): no para
    ins_E = tf.diag(tf.ones([self.ins_len-1]))
    ins_E_up = tf.concat([[[0]*(self.ins_len-1)], ins_E] ,0)
    ins_move_back_matrix = tf.concat([ins_E_up, [[1]]+[[0]]*(self.ins_len-1)], -1)
    cur_ins_addr = ins[2] * tf.matmul(cur_ins_addr, ins_move_back_matrix) +\
              (1-ins[2]) * cur_ins_addr
    #----------------------------------转移到下一条指令---------------------------------------------
    ins_E = tf.diag(tf.ones([self.ins_len-1]))
    ins_E_bottom = tf.concat([ins_E, [[0]*(self.ins_len-1)]] ,0)
    ins_move_matrix = tf.concat([[[0]]*(self.ins_len-1)+[[1]], ins_E_bottom], -1)
    cur_ins_addr = tf.matmul(cur_ins_addr, ins_move_matrix)
    rewards = tf.concat([
        ins[0]*tf.reduce_sum(tf.square(reg_id)*(1-2*prev_state.register_usage),-1,keep_dims=True),
        ins[1]*tf.reduce_sum(tf.square(reg_id)*(2*prev_state.register_usage-1),-1,keep_dims=True),
        ins[2]*tf.ones([tf.shape(reg_id)[0],1])
      ],-1)
    return tf.concat([output,rewards],-1), NPM_State(
        memory=memory,
        register=register,
        register_usage=reg_usage,
        cur_ins_addr=cur_ins_addr,
        cur_mem_addr=cur_mem_addr)
  
  def initial_state(self, batch_size, dtype=tf.float32):
    return NPM_State(
        memory=tf.zeros([batch_size, self.mem_len,self.mem_size], dtype),
        register=tf.zeros([batch_size, self.reg_num,self.reg_size], dtype),
        register_usage=tf.zeros([batch_size, self.reg_num], dtype),
        cur_ins_addr=tf.one_hot(tf.zeros([batch_size],tf.int32), self.ins_len, dtype=dtype),
        cur_mem_addr=tf.one_hot(tf.zeros([batch_size],tf.int32), self.mem_len, dtype=dtype))

  @property
  def state_size(self):
    return NPM_State(
        memory=tf.TensorShape([self.mem_len,self.mem_size]),
        register=tf.TensorShape([self.reg_num,self.reg_size]),
        register_usage=tf.TensorShape([self.reg_num]),
        cur_ins_addr=tf.TensorShape([self.ins_len]),
        cur_mem_addr=tf.TensorShape([self.mem_len]))

  @property
  def output_size(self):
    return tf.TensorShape([self._output_size+self.ins_size])
  
  
