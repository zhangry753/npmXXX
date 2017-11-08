'''
Created on 2017年10月28日

@author: zry
'''
import tensorflow as tf
import sys
import numpy as np

from cellxxx import Neural_programming_machine as NPM

FLAGS = tf.flags.FLAGS
# Model parameters
tf.flags.DEFINE_integer("memory_size", 6, "Size of memory.")
tf.flags.DEFINE_integer("memory_length", 10, "capacity of memory.")
tf.flags.DEFINE_integer("register_size", 6, "Size of register.")
tf.flags.DEFINE_integer("register_num", 4, "number of registers.")
tf.flags.DEFINE_integer("instruction_size", 4, "Size of instruction.")
tf.flags.DEFINE_integer("instruction_length", 20, "length of instruction.")
tf.flags.DEFINE_integer("parameter_size", 20, "Size of parameter.")
tf.flags.DEFINE_integer("parameter_num", 3, "number of parameters for each instruction.")
# Task parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer("min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("max_length", 4,
    "Upper limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 1,
                        "Upper limit on number of copy repeats.")
# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 0.03, "Optimizer learning rate.")
tf.flags.DEFINE_float("learning_rate_L2", 0.001, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")
# Training options.
tf.flags.DEFINE_integer("num_training_iterations", int(1e10),
            "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 500,
            "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", r"checkpoint",
             "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
            "Checkpointing step interval.")
data_max_length = FLAGS.max_length+2 + FLAGS.max_length*FLAGS.max_repeats+1


def run_model(input_sequence, output_size, sequence_length=None):
  """Runs model on input sequence."""
  model_config = {
      "output_size":output_size,
      "memory_size":FLAGS.memory_size,
      "memory_length":FLAGS.memory_length,
      "register_size":FLAGS.register_size,
      "register_num":FLAGS.register_num,
      "instruction_size":FLAGS.instruction_size,
      "instruction_length":FLAGS.instruction_length,
      "parameter_size":FLAGS.parameter_size,
      "parameter_num":FLAGS.parameter_num,
  }
  npm_core = NPM(**model_config)
  initial_state = npm_core.initial_state(FLAGS.batch_size)
  hidden, _ = tf.nn.dynamic_rnn(
      cell=npm_core,
      inputs=input_sequence,
      sequence_length=sequence_length,
      time_major=True,
      initial_state=initial_state)
  return hidden, npm_core


def train(num_training_iterations, report_interval):
  # from DNC-master/train.py
  obs_pattern = tf.cast(
          tf.random_uniform(
              [FLAGS.max_length,FLAGS.batch_size,FLAGS.num_bits], minval=0, maxval=2, dtype=tf.int32),
          tf.float32)
  obs = tf.concat([obs_pattern, tf.zeros([FLAGS.max_length+5,FLAGS.batch_size,FLAGS.num_bits])], 0)
  target = tf.concat([tf.zeros([FLAGS.max_length,FLAGS.batch_size,FLAGS.num_bits]), 
                      obs_pattern, 
                      tf.zeros([5,FLAGS.batch_size,FLAGS.num_bits])], 0)
  output_logits, npm_core = run_model(obs, FLAGS.num_bits)
  # 展示ins和para
  ins_softmax = tf.nn.softmax(npm_core.instructions)
  para_softmax = tf.nn.softmax(npm_core.parameters[:,0,0:FLAGS.register_num])
  ins_sequence = tf.argmax(ins_softmax, -1)
  ins_prob = tf.reduce_max(ins_softmax, -1)
  para_softmax = para_softmax
  inss = npm_core.instructions
  paras = npm_core.parameters[:,0,0:FLAGS.register_num]
  # Used for visualization.
  output = tf.round(tf.sigmoid(output_logits))
  # train loss
  output_loss = tf.reduce_mean(tf.square(target-output_logits),-1)
  output_loss = tf.reduce_sum(output_loss, 0)
  output_loss = tf.reduce_mean(output_loss)
  
  d_ins = tf.stop_gradient(tf.abs(tf.gradients(output_loss, inss)))
  d_para = tf.stop_gradient(tf.abs(tf.gradients(output_loss, npm_core.parameters)))
  L2_norm = -tf.contrib.layers.l2_regularizer(1.)(npm_core.instructions*tf.exp(-d_ins[0])) +\
          -tf.contrib.layers.l2_regularizer(1.)(npm_core.parameters*tf.exp(-d_para[0]))
  L2_loss = tf.clip_by_value(L2_norm, 0., 1e-5)
  train_loss = output_loss + L2_loss
  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)
  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
  # optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
  optimizer = tf.train.AdamOptimizer()
  train_step = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step)
  hooks = []
  # scalar
  tf.summary.scalar('loss', train_loss)
  hooks.append(
      tf.train.SummarySaverHook(
          save_steps=5,
          output_dir=FLAGS.checkpoint_dir+"/logs",
          summary_op=tf.summary.merge_all()))
  # saver
  saver = tf.train.Saver(max_to_keep=50)
  if FLAGS.checkpoint_interval > 0:
    hooks.append(
        tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver))
  # Train.
  with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:
    start_iteration = sess.run(global_step)
    total_loss = 0
    
    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss = sess.run([train_step, train_loss])
      total_loss += loss
      
      if (train_iteration + 1) % report_interval == 0:
        obs_np, output_np, ins_sequence_np, ins_prob_np, para_softmax_np, inss_np, paras_np =\
                sess.run([obs_pattern, output_logits, ins_sequence, ins_prob, para_softmax, inss, paras])
        print("%d: Avg training loss %f.\n  %s\n%s\n  %s\n  %s\n  %s\n"%(
                        train_iteration, total_loss / report_interval,
                        obs_np[:,0,:], output_np[FLAGS.max_length:FLAGS.max_length*2,0,:],
                        # inss_np, paras_np))
                        ins_sequence_np, ins_prob_np, para_softmax_np))
#         print(sess.run(d_ins))
#         print(sess.run(d_para)[0][:,0,0:FLAGS.register_num])
        sys.stdout.flush()
        total_loss = 0


def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
#   sys.stdout = open(r'message.log','w')
  tf.app.run()





