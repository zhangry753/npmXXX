'''
Created on 2017年11月17日

@author: zry
'''
# Model parameters
# Size of memory.
memory_size=10 
# capacity of memory.
memory_length = 10 
# Size of register.
register_size = 10 
# number of registers.
register_num = 10 
# Size of instruction.
instruction_size = 3 
# length of instruction.
instruction_length = None # 2*max_length+5 
# Size of parameter.
parameter_size = 20 
# number of parameters for each instruction.
parameter_num = 3

# Task parameters
# Batch size for training.
batch_size = 30 
# Dimensionality of each vector to copy
num_bits = 10 
#Lower limit on number of vectors in the observation pattern to copy
min_length = 1 
# Upper limit on number of vectors in the observation pattern to copy
max_length = 10
# Lower limit on number of copy repeats.
min_repeats = 1
# Upper limit on number of copy repeats.                        
max_repeats = 1
                        
# Optimizer parameters.
# Gradient clipping norm limit.
max_grad_norm = 50 
# Optimizer learning rate.
learning_rate = 0.03 
# Optimizer learning rate.
learning_rate_L2 = 0.001 
# Epsilon used for RMSProp optimizer.
optimizer_epsilon = 1e-10
                    
# Training options.
# Number of iterations to train for.
num_training_iterations = int(1e10)
# Iterations between reports (samples, valid loss).            
report_interval = 200
# Checkpointing directory.            
checkpoint_dir = r"checkpoint"
# Checkpointing step interval.             
checkpoint_interval = -1




instruction_length = max_length*2+5



