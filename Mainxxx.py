'''
Created on 2017年10月28日

@author: zry
'''
import tensorflow as tf
import sys
import numpy as np

from cellxxx import Neural_programming_machine as NPM
import config
from dqn import DQN


def cosine_distance(a,b):
  _EPSILON = 1e-6
  dot = tf.reduce_sum(a*b,-1,keep_dims=True)
  a_norms = tf.sqrt(tf.reduce_sum(a * a, axis=-1,keep_dims=True)+ _EPSILON)
  b_norms = tf.sqrt(tf.reduce_sum(b * b, axis=-1,keep_dims=True)+ _EPSILON)
  similarity = dot / (a_norms*b_norms + _EPSILON)
  similarity = 2*similarity-1
  return similarity

def run_model(input_sequence, output_size, sequence_length=None):
  """Runs model on input sequence."""
  model_config = {
      "output_size":output_size,
      "memory_size":config.memory_size,
      "memory_length":config.memory_length,
      "register_size":config.register_size,
      "register_num":config.register_num,
      "instruction_size":config.instruction_size,
      "instruction_length":config.instruction_length,
      "parameter_size":config.parameter_size,
      "parameter_num":config.parameter_num,
  }
  npm_core = NPM(**model_config)
  initial_state = npm_core.initial_state(config.batch_size)
  rnn_outputs, _ = tf.nn.dynamic_rnn(
      cell=npm_core,
      inputs=input_sequence,
      sequence_length=sequence_length,
      time_major=True,
      initial_state=initial_state)
  index = 0
  hidden = rnn_outputs[:, :, index: config.num_bits]
  index += config.num_bits
  register = rnn_outputs[:, :, index: index+config.register_num*config.register_size]
  index += config.register_num*config.register_size
  reg_usage = rnn_outputs[:, :, index: index+config.register_num]
  index += config.register_num
  return hidden, npm_core, register, reg_usage


def run_dqn(register,reg_usage,target,instructions,parameters,rewards):
  ins = instructions[:,0:config.instruction_size-1]
  para = parameters[:,0,:]
  actions_3D = tf.matmul(tf.expand_dims(ins,2), tf.expand_dims(para,1))
  actions = tf.reshape(actions_3D, [-1,(config.instruction_size-1)*config.parameter_size])
  terminals = instructions[:,config.instruction_size-1:]
  dqn_core = DQN(tf.stop_gradient(register),
      tf.stop_gradient(reg_usage),
      tf.stop_gradient(target),
      tf.stop_gradient(actions),
      tf.stop_gradient(rewards),
      tf.stop_gradient(terminals))
  return dqn_core


def calculate_rewards(instructions, reg_id, reg_usage, output, target):
  reg_id = tf.expand_dims(reg_id, 1)
  ins_all = [tf.expand_dims(tf.expand_dims(ins,-1),-1) for ins in tf.unstack(instructions,axis=1)]
  terminals = instructions[:,config.instruction_size-1:]
  terminals_prod = tf.cumprod(1-terminals, axis=0, exclusive=True)
#   pred_similar = cosine_distance(output, target)
  pred_similar = tf.reduce_mean(tf.square(target-output),-1, keep_dims=True)
  pred_similar = 2 * tf.clip_by_value(pred_similar, 1e-5, 1.) - 1
  reg_usage = tf.stop_gradient(reg_usage)
  rewards_each_act = tf.concat([
      ins_all[0] * reg_id*(1-2*reg_usage),
      ins_all[1] * pred_similar,
      ins_all[2] * tf.ones([1,config.batch_size,1])*0.05
    ], -1)
  rewards = terminals_prod * tf.reduce_max(rewards_each_act, -1)
  return rewards, rewards_each_act


def train(num_training_iterations, report_interval):
  obs_pattern = tf.cast(
          tf.random_uniform(
              [config.max_length,config.batch_size,config.num_bits], minval=0, maxval=2, dtype=tf.int32),
          tf.float32)
  obs = tf.concat([obs_pattern, tf.zeros([config.max_length+5,config.batch_size,config.num_bits])], 0)
  target = tf.concat([tf.zeros([config.max_length,config.batch_size,config.num_bits]), 
                      obs_pattern, 
                      tf.zeros([5,config.batch_size,config.num_bits])], 0)
  output_logits, npm_core, register, reg_usage = run_model(obs, config.num_bits)
  # 展示ins和para
  ins_softmax = tf.nn.softmax(npm_core.instructions)
  para_softmax = tf.nn.softmax(npm_core.parameters)
  reg_id = tf.nn.softmax(npm_core.parameters)[:,0,0:config.register_num]
  ins_seq = tf.argmax(ins_softmax, -1)
  ins_prob = tf.reduce_max(ins_softmax, -1)
  regid_seq = tf.argmax(reg_id, -1)
  regid_prob = tf.reduce_max(reg_id, -1)
  # rewards
  rewards, rewards_each_act = calculate_rewards(ins_softmax, reg_id, reg_usage, output_logits, target)
  rewards_bat = tf.reduce_mean(rewards, 0)
  rewards_loss = 1 - tf.reduce_mean(rewards_bat)
  # train loss
  output_loss_each = tf.reduce_mean(tf.square(target-output_logits),-1)
#   output_loss_each = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output_logits)
  output_loss = tf.reduce_mean(output_loss_each)
  train_loss = rewards_loss
  d_ins = tf.gradients(output_loss, npm_core.instructions)
  d_para = tf.gradients(output_loss, npm_core.parameters)
  d_ins2 = tf.gradients(rewards_loss, npm_core.instructions)
  d_para2 = tf.gradients(rewards_loss, npm_core.parameters)
  # DQN
  dqn_core = run_dqn(register, reg_usage, target, ins_softmax, para_softmax, rewards)
  # from DNC-master/train.py
  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(train_loss, trainable_variables), config.max_grad_norm)
  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
  optimizer = tf.train.RMSPropOptimizer(config.learning_rate, epsilon=config.optimizer_epsilon)
#   optimizer = tf.train.AdamOptimizer()
  train_step = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step)
  hooks = []
  # scalar
#   tf.summary.scalar('loss', train_loss)
#   hooks.append(
#       tf.train.SummarySaverHook(
#           save_steps=5,
#           output_dir=config.checkpoint_dir+"/logs",
#           summary_op=tf.summary.merge_all()))
  # saver
  saver = tf.train.Saver(max_to_keep=50)
  if config.checkpoint_interval > 0:
    hooks.append(
        tf.train.CheckpointSaverHook(
            checkpoint_dir=config.checkpoint_dir,
            save_steps=config.checkpoint_interval,
            saver=saver))
  # Train.
  with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=config.checkpoint_dir) as sess:
    start_iteration = sess.run(global_step)
    total_loss = 0
    total_dqn_loss = 0
    
    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss = sess.run([train_step, train_loss])
      total_loss += loss
#       if train_iteration>1000:
#         _, dqn_loss_val = sess.run([dqn_core.train_step, dqn_core.cost])
#         total_dqn_loss += dqn_loss_val
      
      if (train_iteration + 1) % report_interval == 0:
        obs_np, output_np, ins_sequence_np, ins_prob_np, regid_seq_np, regid_prob_np, reg_usage_np, rewards_np, register_np =\
                sess.run([obs_pattern, output_logits, 
                          ins_seq, ins_prob, regid_seq, regid_prob, 
                          reg_usage, rewards_each_act, register])
        print("%d: Avg training loss %f.\t%f.\n "%(
                        train_iteration, total_loss/report_interval, total_dqn_loss/report_interval))
#         print(obs_np[:,0,:], output_np[config.max_length:config.max_length*2,0,:], sep='\n')
        print(ins_sequence_np, ins_prob_np, regid_seq_np, regid_prob_np, sep='\n')
#         print(register_np[:,0,:])
#         print(reg_usage_np[:,0,:])
#         print(rewards_np[:,0,:])
#         if train_iteration>1000:
#         print(sess.run(dqn_core.y)[:,0,:])
#           print(sess.run(dqn_core.ins_pred))
#           print(sess.run(dqn_core.para_pred))
#         print(sess.run(d_ins)[0][:config.max_length*2+1,:])
#         print(sess.run(d_para)[0][:config.max_length*2+1,0,0:config.register_num])
        print(sess.run(d_ins2)[0][:config.max_length*2+1,:])
        print(sess.run(d_para2)[0][:config.max_length*2+1,0,0:config.register_num])
        sys.stdout.flush()
        total_loss = 0
        total_dqn_loss = 0


def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train(config.num_training_iterations, config.report_interval)


if __name__ == "__main__":
  sys.stdout = open(r'message.log','w')
  tf.app.run()





