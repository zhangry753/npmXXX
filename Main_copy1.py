'''
Created on 2017年10月28日

@author: zry
'''
import tensorflow as tf
import sys

def encode(input_, target_shape):
    input_ = tf.cast(input_, tf.float32)
    zeros = tf.zeros([target_shape - tf.shape(input_)[-1]])
    while not input_.get_shape().most_specific_compatible_shape(zeros.get_shape()):
        zeros = tf.expand_dims(zeros, 0)
    return  tf.concat([input_, zeros], -1)
def decode(input_, target_shape):
    return tf.reshape(input_[0:target_shape], [1,target_shape])
    
    
class AI_coding():
    def __init__(self, input_size, input_length, memory_size, memory_length, register_size, register_num,
                 instruction_size, instruction_length, parameter_size, parameter_num, input_matrix):
        self.input_size = input_size
        self.input_len = input_length
        self.mem_size = memory_size
        self.mem_len = memory_length
        self.reg_size = register_size
        self.reg_num = register_num
        self.ins_size = instruction_size
        self.ins_len = instruction_length
        self.para_size = parameter_size
        self.para_num = parameter_num
        self.input = input_matrix
        self.memory = tf.constant([[0]*self.mem_size]*self.mem_len, dtype=tf.float32, name="memory")
        self.register = tf.constant([[0]*self.reg_size]*self.reg_num, dtype=tf.float32, name="register")
        self.cur_ins_addr = tf.constant([[1]+[0]*(self.ins_len-1)], dtype=tf.float32, name="cur_ins_addr")
        self.cur_mem_addr = tf.constant([[1]+[0]*(self.mem_len-1)], dtype=tf.float32, name="cur_mem_addr")
    
    def run_step(self, ins, para):
#         ins = tf.nn.relu((ins-0.1)/0.9)
#         para = tf.nn.relu((para-0.1)/0.9)
        #----------------------------------更新寄存器---------------------------------------------
        # ins0 常量到寄存器: para0--寄存器id,para1--常量
        constant_num = decode(para[1],1)
        new_reg_value = ins[0] * encode(constant_num, self.reg_size)
        # ins1 输入到寄存器: para0--寄存器id,para1--输入id
        input_id = decode(para[1], self.input_len)
        input_value = tf.matmul(input_id, self.input)
        new_reg_value += ins[1] * encode(input_value, self.reg_size)
        # ins2 内存到寄存器: para0--寄存器id,para1--内存地址
        mem_addr = decode(para[1], self.mem_len)
        mem_value = tf.matmul(mem_addr, self.memory)
        new_reg_value += ins[2] * encode(mem_value, self.reg_size)
        # ins3 加法: para0--结果寄存器id,para1--数1寄存器id,para2--数2寄存器id
        num1_reg_id = decode(para[1], self.reg_num)
        num1 = tf.matmul(num1_reg_id, self.register)
        num2_reg_id = decode(para[2], self.reg_num)
        num2 = tf.matmul(num2_reg_id, self.register)
        new_reg_value +=ins[3] * (num1 + num2)
        # 更新寄存器
        register_id = decode(para[0],self.reg_num)
        register_id_neg = tf.diag(tf.reshape(1-register_id, [self.reg_num]))
        new_reg_value_all = tf.matmul(register_id, new_reg_value, transpose_a=True) + tf.matmul(register_id_neg, self.register)
        self.register = (ins[0]+ins[1]+ins[2]+ins[3])*new_reg_value_all + (1-ins[0]-ins[1]-ins[2]-ins[3])*self.register
        #----------------------------------更新内存---------------------------------------------
        # ins4 开辟空间:  no para 返回内存地址
        new_mem_addr = self.cur_mem_addr
        mem_E = tf.diag(tf.ones([self.mem_len-1]))
        mem_E_bottom = tf.concat([mem_E,[[0]*(self.mem_len-1)]],0)
        mem_move_matrix = tf.concat([[[0]]*(self.mem_len-1)+[[1]], mem_E_bottom], -1)
        mem_stay_matrix = tf.diag(tf.ones([self.mem_len]))
        self.cur_mem_addr = ins[4]*tf.matmul(self.cur_mem_addr,mem_move_matrix) +\
                            (1-ins[4])*tf.matmul(self.cur_mem_addr,mem_stay_matrix)
        # ins5 存入内存: para0--寄存器id,para1--内存地址
        reg_id = decode(para[0], self.reg_num)
        mem_addr = decode(para[1], self.mem_len)
        new_mem_value = encode(tf.matmul(reg_id, self.register), self.mem_size)
        self.memory += ins[5]*tf.matmul(mem_addr, new_mem_value, transpose_a=True)
        #----------------------------------流程控制---------------------------------------------
        # ins6 小于跳转(任意para0<para2则跳转): para0--寄存器1id,para1--寄存器2id,para2--指令地址
        num1_reg_id = decode(para[0], self.reg_num)
        num1 = tf.matmul(num1_reg_id, self.register)
        num2_reg_id = decode(para[1], self.reg_num)
        num2 = tf.matmul(num2_reg_id, self.register)
        jump_prob = ins[6] * tf.nn.sigmoid(1000*(num2-num1+0.01))      #判断精度为sigmoid(x)≈1的x解，大约0.5
#         jump_prob = tf.reduce_max(jump_prob)
        jump_prob = jump_prob[0][0]
        new_ins_addr = decode(para[2], self.ins_len)
        ins_stay_matrix = tf.diag(tf.ones([self.ins_len]))
        self.cur_ins_addr = jump_prob * new_ins_addr +\
                            (1-jump_prob) * tf.matmul(self.cur_ins_addr, ins_stay_matrix)
        # ins7 输出: para0--寄存器id
        reg_id = decode(para[0], self.reg_num)
        reg_value = tf.matmul(reg_id,self.register)
        output = ins[7]*reg_value
        # ins8 终止(重复执行本条指令): no para
        ins_E = tf.diag(tf.ones([self.ins_len-1]))
        ins_E_up = tf.concat([[[0]*(self.ins_len-1)], ins_E] ,0)
        ins_move_back_matrix = tf.concat([ins_E_up, [[1]]+[[0]]*(self.ins_len-1)], -1)
        ins_stay_matrix = tf.diag(tf.ones([self.ins_len]))
        self.cur_ins_addr = ins[8] * tf.matmul(self.cur_ins_addr, ins_move_back_matrix) +\
                            (1-ins[8]) * tf.matmul(self.cur_ins_addr, ins_stay_matrix)
                            
#         print(sess.run(tf.matmul([[1.,0.]],self.register)), end='\t')
#         print(sess.run(tf.matmul([[0.,1.]],self.register)), end='\t')
#         print(sess.run(jump_prob), end='\t')
#         print(sess.run(output))
        return output, new_mem_addr
    
    def run(self, instructions, parameters):
        x_input = tf.placeholder(tf.float32, [2,3], "x_input")
        correct_label = tf.placeholder(tf.float32, [20,5], "correct_label")
        ins_E = tf.diag(tf.ones([self.ins_len-1]))
        ins_E_bottom = tf.concat([ins_E, [[0]*(self.ins_len-1)]] ,0)
        ins_move_matrix = tf.concat([[[0]]*(self.ins_len-1)+[[1]], ins_E_bottom], -1)
        output_list = []
        for _ in range(20):
            cur_ins = tf.matmul(self.cur_ins_addr, instructions)
            cur_ins = tf.reshape(cur_ins, [self.ins_size])
            parameters_flat = tf.reshape(parameters, [self.ins_len, self.para_num*self.para_size])
            cur_para = tf.matmul(self.cur_ins_addr, parameters_flat)
            cur_para = tf.reshape(cur_para, [self.para_num,self.para_size])
            cur_output, new_mem_addr = self.run_step(cur_ins, cur_para)
            output_list.append(cur_output)
            self.cur_ins_addr = tf.matmul(self.cur_ins_addr, ins_move_matrix)
        
        predicts = tf.concat(output_list, 0)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=correct_label, logits=predicts))
#         cross_entropy = tf.reduce_sum(correct_label * tf.log(correct_label/predicts), axis=1)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.03).minimize(loss)
        return x_input, correct_label, predicts, optimizer, loss



if __name__ == "__main__":
#     sys.stdout = open(r'C:\Users\zry\Desktop\message.log', "w")
    
    X = tf.constant([[1,1,1],[3]], dtype=tf.float32, name="input")
    correct_label = [[0]*5]*4 + [[1]+[0]*4] + [[0]*5]*6 + [[1]+[0]*4] + [[0]*5]*6 + [[1]+[0]*4] + [[0]*5]
    correct_label = tf.constant(correct_label, tf.float32)
    input_size = 1
    input_length = 2
    memory_size = 5
    memory_length = 3
    register_size = 5
    register_num = 2
    instruction_size = 9
    parameter_size = 15
    parameter_num = 3
    instructions = tf.Variable([[0],[4],[5],[1],[7],[2],[0],[3],[5],[1],[6],[8]], name="instructions")
    instruction_length = instructions.get_shape()[0].value
    instructions = tf.Variable(tf.squeeze(tf.one_hot(instructions, instruction_size, dtype=tf.float32)))
    parameters = tf.Variable([[encode([1,0], parameter_size), encode([0], parameter_size), [0]*parameter_size],
                             [[0]*parameter_size, [0]*parameter_size, [0]*parameter_size],
                             [encode([1,0], parameter_size), encode([1]+[0]*(memory_length-1), parameter_size), [0]*parameter_size],
                             [encode([1,0], parameter_size), encode([1,0], parameter_size), [0]*parameter_size],
                             [encode([1,0], parameter_size), [0]*parameter_size, [0]*parameter_size],
                             [encode([1,0], parameter_size), encode([1]+[0]*(memory_length-1), parameter_size), [0]*parameter_size],
                             [encode([0,1], parameter_size), encode([1], parameter_size), [0]*parameter_size],
                             [encode([1,0], parameter_size), encode([0,1], parameter_size), encode([1,0], parameter_size)],
                             [encode([1,0], parameter_size), encode([1]+[0]*(memory_length-1), parameter_size), [0]*parameter_size],
                             [encode([0,1], parameter_size), encode([0,1], parameter_size), [0]*parameter_size],
                             [encode([1,0], parameter_size), encode([0,1], parameter_size), encode([0]*2+[1]+[0]*(instruction_length-3), parameter_size)],
                             [[0]*parameter_size, [0]*parameter_size, [0]*parameter_size]
                             ], tf.float32, name="parameters")
    
    # 加入误差
    instructions = instructions*4-4
    instructions = tf.nn.softmax(instructions)
    parameters = parameters*4-4
    parameters = tf.nn.softmax(parameters)
    model = AI_coding(input_size,input_length,memory_size,memory_length,register_size,register_num,
                      instruction_size,instruction_length,parameter_size,parameter_num,X)
    x_input, label_input, predicts, optimizer,cross_entropy = model.run(instructions, parameters)
    feed_dict = {x_input:X, label_input:correct_label}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = -1
        while(True):
            step += 1
            sess.run(optimizer, feed_dict=feed_dict)
            if step%1000==0:
#                 summary_str = sess.run(merge_op)
#                 summary_writer.add_summary(summary_str, step)
                print(sess.run(cross_entropy), feed_dict=feed_dict)
                print(sess.run(tf.reduce_max(predicts,-1)), feed_dict=feed_dict)
                print(sess.run(tf.argmax(instructions)))
                print(sess.run(tf.reduce_max(instructions,-1)))
                print(sess.run(tf.reduce_max(parameters,-1)))
#                 sys.stdout.flush()


