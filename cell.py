'''
Created on 2017年10月31日

@author: zry
'''
import collections
import sonnet as snt
import tensorflow as tf

NPM_State = collections.namedtuple('AIPState', ('memory', 'register','cur_ins_addr','cur_mem_addr'))

def onehot_max(input_, min_value=0.8):
  input_softmax = tf.nn.softmax(input_)
  condition = tf.reduce_max(input_, -1) >= min_value
  input_onehot = tf.one_hot(tf.argmax(input_softmax, -1), input_.get_shape()[-1].value)
  return tf.where(condition, input_onehot, input_softmax)

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
    self.instructions = tf.get_variable("instructions", [self.ins_len, self.ins_size])
    self.parameters = tf.get_variable("parameters", [self.ins_len, self.para_num, self.para_size])
#     self.instructions = tf.Variable(tf.random_normal([self.ins_len, self.ins_size]), name='instructions')
#     self.parameters = tf.Variable(tf.zeros([self.ins_len, self.para_num, self.para_size]), name='parameters')
#     self.instructions = tf.reshape(tf.one_hot([[9],[1],[1],[1],[9],[7],[7],[7],[7],[7],[7]], self.ins_size), [self.ins_len,self.ins_size])
#     self.instructions = tf.Variable(self.instructions, trainable=False)
#     self.parameters = tf.constant([
#         [[0]*self.para_size,[0]*self.para_size,[0]*self.para_size],
#         [[1]+[0]*(self.para_size-1),[0]*self.para_size,[0]*self.para_size],
#         [[0,1]+[0]*(self.para_size-2),[0]*self.para_size,[0]*self.para_size],
#         [[0,0,1]+[0]*(self.para_size-3),[0]*self.para_size,[0]*self.para_size],
#         [[0]*self.para_size,[0]*self.para_size,[0]*self.para_size],
#         [[1]+[0]*(self.para_size-1),[0]*self.para_size,[0]*self.para_size],
#         [[0,1]+[0]*(self.para_size-2),[0]*self.para_size,[0]*self.para_size],
#         [[0,0,1]+[0]*(self.para_size-3),[0]*self.para_size,[0]*self.para_size],
#         [[1]+[0]*(self.para_size-1),[0]*self.para_size,[0]*self.para_size],
#         [[0,1]+[0]*(self.para_size-2),[0]*self.para_size,[0]*self.para_size],
#         [[0,0,1]+[0]*(self.para_size-3),[0]*self.para_size,[0]*self.para_size],
#     ], tf.float32)
    self.linears = {
        "para_to_reg":snt.Linear(self.reg_size, name="para_to_reg"),
        "input_to_reg":snt.Linear(self.reg_size, name="input_to_reg"),
        "para_to_mem_addr":snt.Linear(self.mem_len, name="para_to_mem_addr"),
        "mem_to_reg":snt.Linear(self.reg_size, name="mem_to_reg"),
        "para_to_reg_id":snt.Linear(self.reg_num, name="para_to_reg_id"),
        "reg_to_reg_add":snt.Linear(self.reg_size, name="reg_to_reg_add"),
        "reg_to_mem":snt.Linear(self.mem_size, name="reg_to_mem"),
        "para_to_ins_addr":snt.Linear(self.ins_len, name="para_to_ins_addr"),
        "reg_to_prob_jump":snt.Linear(1, name="reg_to_prob_jump"),
        "reg_to_output":snt.Linear(self._output_size,use_bias=False, name="reg_to_output"),
      }
    # 读取权值
#     reader = tf.train.NewCheckpointReader(r'C:\Users\zry\Desktop\checkpoint\model.ckpt-10000')
#     for tensor_name,linear in self.linears.items():
#       tensor_name_w = "AI_programming/"+tensor_name+"/w"
#       linear._w = reader.get_tensor(tensor_name_w)
#       if not tensor_name == "reg_to_output":
#         tensor_name_b = "AI_programming/"+tensor_name+"/b"
#         linear._b = reader.get_tensor(tensor_name_b)
        
    
  def _build(self, inputs, prev_state):
    instructions_onehot = onehot_max(self.instructions, 0)
    memory=prev_state.memory
    register=prev_state.register
    cur_ins_addr=prev_state.cur_ins_addr
    cur_mem_addr=prev_state.cur_mem_addr
    
    cur_ins = tf.matmul(cur_ins_addr, instructions_onehot)
    parameters_flat = tf.reshape(self.parameters, [self.ins_len, self.para_num*self.para_size])
    cur_para = tf.reshape(tf.matmul(cur_ins_addr, parameters_flat), [-1,self.para_num,self.para_size])
    ins = [tf.expand_dims(item,1) for item in tf.unstack(cur_ins, axis=1)]
    para = tf.unstack(cur_para, axis=1)
    #----------------------------------更新寄存器---------------------------------------------
    # ins0 常量到寄存器: para0--寄存器id,para1--常量
    constant_num = self.linears["para_to_reg"](para[1])
    new_reg_value = ins[0] * constant_num
    # ins1 输入到寄存器: para0--寄存器id
    input_value = self.linears["input_to_reg"](inputs)
    new_reg_value += ins[1] * input_value
    # ins2 内存到寄存器: para0--寄存器id,para1--内存地址
    mem_addr = tf.nn.softmax(self.linears["para_to_mem_addr"](para[1]))
    mem_value = tf.reshape(tf.matmul(tf.expand_dims(mem_addr,1), memory), [-1,self.mem_size])
    mem_value = self.linears["mem_to_reg"](mem_value)
    new_reg_value += ins[2] * mem_value
    # ins3 加法: para0--结果寄存器id,para1--数1寄存器id,para2--数2寄存器id
    num1_reg_id = tf.nn.softmax(self.linears["para_to_reg_id"](para[1]))
    num1 = tf.reshape(tf.matmul(tf.expand_dims(num1_reg_id,1), register), [-1,self.reg_size])
    num2_reg_id = tf.nn.softmax(self.linears["para_to_reg_id"](para[2]))
    num2 = tf.reshape(tf.matmul(tf.expand_dims(num2_reg_id,1), register), [-1,self.reg_size])
    add_value = self.linears["reg_to_reg_add"](num1 + num2)
    new_reg_value += ins[3] * add_value
    # 更新寄存器
    register_id = tf.nn.softmax(self.linears["para_to_reg_id"](para[0]))
    new_reg_value_all = tf.nn.tanh(tf.matmul(tf.expand_dims(register_id,2), tf.expand_dims(new_reg_value,1)) +\
            tf.expand_dims(1-register_id,2) * register)
    register = tf.expand_dims(ins[0]+ins[1]+ins[2]+ins[3],1) * new_reg_value_all  +\
            tf.expand_dims(1-ins[0]-ins[1]-ins[2]-ins[3],1) * register
    #----------------------------------更新内存---------------------------------------------
    # ins4 开辟空间:  no para 返回内存地址
    new_mem_addr = cur_mem_addr
    mem_E = tf.diag(tf.ones([self.mem_len-1]))
    mem_E_bottom = tf.concat([mem_E,[[0]*(self.mem_len-1)]],0)
    mem_move_matrix = tf.concat([[[0]]*(self.mem_len-1)+[[1]], mem_E_bottom], -1)
    next_mem_addr = tf.matmul(cur_mem_addr, mem_move_matrix)
    cur_mem_addr = ins[4]*next_mem_addr + (1-ins[4])*cur_mem_addr
    cur_mem_addr = cur_mem_addr / tf.reduce_sum(cur_mem_addr, -1, keep_dims=True)
    # ins5 存入内存: para0--寄存器id,para1--内存地址
    reg_id = tf.nn.softmax(self.linears["para_to_reg_id"](para[0]))
    mem_addr = tf.expand_dims(tf.nn.softmax(self.linears["para_to_mem_addr"](para[1])), 2)
    reg_value = tf.reshape(tf.matmul(tf.expand_dims(reg_id,1), register), [-1,self.reg_size])
    new_mem_value = tf.reshape(tf.nn.tanh(self.linears["reg_to_mem"](reg_value)), [-1,1,self.mem_size])
    memory += tf.expand_dims(ins[5],1) * tf.matmul(mem_addr, new_mem_value)
    #----------------------------------流程控制---------------------------------------------
    # ins6 跳转: para0--指令地址,para1--寄存器1id,para2--寄存器2id
    num1_reg_id = tf.nn.softmax(self.linears["para_to_reg_id"](para[1]))
    num1 = tf.reshape(tf.matmul(tf.expand_dims(num1_reg_id,1), register), [-1,self.reg_size])
    num2_reg_id = tf.nn.softmax(self.linears["para_to_reg_id"](para[2]))
    num2 = tf.reshape(tf.matmul(tf.expand_dims(num2_reg_id,1), register), [-1,self.reg_size])
    jump_prob = tf.nn.sigmoid(self.linears["reg_to_prob_jump"](tf.concat([num1,num2], -1)))
    jump_prob = ins[6] * jump_prob
    new_ins_addr = tf.nn.softmax(self.linears["para_to_ins_addr"](para[0]))
    cur_ins_addr = jump_prob*new_ins_addr + (1-jump_prob)*cur_ins_addr
    # ins7 输出: para0--寄存器id
    reg_id = tf.nn.softmax(self.linears["para_to_reg_id"](para[0]))
    reg_value = tf.reshape(tf.matmul(tf.expand_dims(reg_id,1), register), [-1,self.reg_size])
    output = ins[7] * self.linears["reg_to_output"](reg_value)
    # ins8 终止(重复执行本条指令): no para
    ins_E = tf.diag(tf.ones([self.ins_len-1]))
    ins_E_up = tf.concat([[[0]*(self.ins_len-1)], ins_E] ,0)
    ins_move_back_matrix = tf.concat([ins_E_up, [[1]]+[[0]]*(self.ins_len-1)], -1)
    cur_ins_addr = ins[8] * tf.matmul(cur_ins_addr, ins_move_back_matrix) +\
              (1-ins[8]) * cur_ins_addr
    # ins9 跳过(什么都不做): no para
    #----------------------------------转移到下一条指令---------------------------------------------
    cur_ins_addr = cur_ins_addr / tf.reduce_sum(cur_ins_addr, -1, keep_dims=True)
    ins_E = tf.diag(tf.ones([self.ins_len-1]))
    ins_E_bottom = tf.concat([ins_E, [[0]*(self.ins_len-1)]] ,0)
    ins_move_matrix = tf.concat([[[0]]*(self.ins_len-1)+[[1]], ins_E_bottom], -1)
    cur_ins_addr = tf.matmul(cur_ins_addr, ins_move_matrix)
    return output, NPM_State(
        memory=memory,
        register=register,
        cur_ins_addr=cur_ins_addr,
        cur_mem_addr=cur_mem_addr)
  
  def initial_state(self, batch_size, dtype=tf.float32):
    return NPM_State(
        memory=tf.zeros([batch_size, self.mem_len,self.mem_size], dtype),
        register=tf.zeros([batch_size, self.reg_num,self.reg_size], dtype),
        cur_ins_addr=tf.one_hot(tf.zeros([batch_size],tf.int32), self.ins_len, dtype=dtype),
        cur_mem_addr=tf.one_hot(tf.zeros([batch_size],tf.int32), self.mem_len, dtype=dtype))

  @property
  def state_size(self):
    return NPM_State(
        memory=tf.TensorShape([self.mem_len,self.mem_size]),
        register=tf.TensorShape([self.reg_num,self.reg_size]),
        cur_ins_addr=tf.TensorShape([self.ins_len]),
        cur_mem_addr=tf.TensorShape([self.mem_len]))

  @property
  def output_size(self):
    return tf.TensorShape([self._output_size])
  
  
