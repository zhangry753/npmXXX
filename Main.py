'''
Created on 2017年10月28日

@author: zry
'''
import tensorflow as tf
import sonnet as snt
import repeat_copy
import sys

from cell import Neural_programming_machine as NPM

FLAGS = tf.flags.FLAGS
# Model parameters
tf.flags.DEFINE_integer("memory_size", 5, "Size of memory.")
tf.flags.DEFINE_integer("memory_length", 10, "capacity of memory.")
tf.flags.DEFINE_integer("register_size", 5, "Size of register.")
tf.flags.DEFINE_integer("register_num", 4, "number of registers.")
tf.flags.DEFINE_integer("instruction_size", 10, "Size of instruction.")
tf.flags.DEFINE_integer("instruction_length", 20, "length of instruction.")
tf.flags.DEFINE_integer("parameter_size", 20, "Size of parameter.")
tf.flags.DEFINE_integer("parameter_num", 3, "number of parameters for each instruction.")
# Task parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer("min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("max_length", 5,
    "Upper limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 5,
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
tf.flags.DEFINE_integer("report_interval", 100,
            "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", r"checkpoint",
             "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
            "Checkpointing step interval.")
data_max_length = FLAGS.max_length+2 + FLAGS.max_length*FLAGS.max_repeats+1


def run_model(input_sequence, output_size, sequence_length=[data_max_length]*FLAGS.batch_size):
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
#   conv_weight = tf.Variable(tf.random_normal([3, output_size, output_size]), name='conv_w')
#   output_sequence = tf.nn.conv1d(tf.transpose(hidden,[1,0,2]), conv_weight, 1, 'SAME')
#   output_sequence = tf.transpose(output_sequence, [1,0,2])
  ins_softmax = tf.nn.softmax(npm_core.instructions)
  hidden_trans = tf.transpose(hidden, [1,0,2])
  hidden_flat = tf.reshape(hidden_trans, [FLAGS.batch_size, data_max_length*(FLAGS.num_bits+1)])
  output_flat = snt.Linear(data_max_length*(FLAGS.num_bits+1), name="output_format")(hidden_flat)
  output = tf.transpose(tf.reshape(output_flat, [FLAGS.batch_size, data_max_length, FLAGS.num_bits+1]), [1,0,2])
  _,ins_variance = tf.nn.moments(ins_softmax, -1)
  ins_L2_norm = -tf.reduce_mean(ins_variance)
#   ins_L2_norm = tf.contrib.layers.l2_regularizer(FLAGS.learning_rate_L2)(ins_softmax)
  return hidden, ins_L2_norm, tf.argmax(ins_softmax, -1), tf.reduce_max(ins_softmax, -1)


def train(num_training_iterations, report_interval):
  # from DNC-master/train.py
  dataset = repeat_copy.RepeatCopy(FLAGS.num_bits, FLAGS.batch_size,
                                   FLAGS.min_length, FLAGS.max_length,
                                   FLAGS.min_repeats, FLAGS.max_repeats)
  dataset_tensors = dataset()
  data_length = tf.shape(dataset_tensors.observations)[0]
  zeros_obs = tf.zeros([data_max_length - data_length, FLAGS.batch_size, FLAGS.num_bits+2])
  obs_new = tf.concat([dataset_tensors.observations, zeros_obs], 0)
  output_logits, ins_L2_norm, ins_sequence, ins_prob =\
          run_model(obs_new, dataset.target_size)
  output_logits = output_logits[:data_length,:,:]
  # Used for visualization.
  output = tf.round(
      tf.expand_dims(dataset_tensors.mask, -1) * tf.sigmoid(output_logits))
 
  train_loss = dataset.cost(output_logits, dataset_tensors.target,
                            dataset_tensors.mask)
  train_loss += tf.where(train_loss<10,1.,0.)*ins_L2_norm
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
#   optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
  optimizer = tf.train.AdamOptimizer()
  tf.train.AdamOptimizer
  train_step = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step)
  # saver
  saver = tf.train.Saver(max_to_keep=50)
  if FLAGS.checkpoint_interval > 0:
    hooks = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver)
    ]
  else:
    hooks = []
  # Train.
  with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:
    start_iteration = sess.run(global_step)
    total_loss = 0
 
    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss = sess.run([train_step, train_loss])
      total_loss += loss
       
      if (train_iteration + 1) % report_interval == 0:
        dataset_tensors_np, output_np, ins_sequence_np, ins_prob_np =\
                sess.run([dataset_tensors, output, ins_sequence, ins_prob])
        dataset_string = dataset.to_human_readable(dataset_tensors_np,output_np)
        print("%d: Avg training loss %f.  %s\n  %s\n  %s"%(
                        train_iteration, total_loss / report_interval,
                        dataset_string, ins_sequence_np, ins_prob_np))
        sys.stdout.flush()
        total_loss = 0


def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
#   sys.stdout = open(r'message.log','w')
  tf.app.run()





