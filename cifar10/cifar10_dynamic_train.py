# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import decomposition
import cifar10
import resource
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'cifar10_dynamic_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_boolean('stable', False,
                            """Training stop flag.""")
tf.app.flags.DEFINE_boolean('dropout_prob', 0.5,
                            """Training stop flag.""")
tf.app.flags.DEFINE_boolean('dropout_prob2', 0.5,
                            """Training stop flag.""")

def dynamic_train():
  start = True
  steps = 0
  parameters = {}
  max_training_steps = 400000
  while steps < max_training_steps:
    print("training cifar10 in steps: %d" % steps)
    # r0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print ("@test mem:%dM" % (r0/1024))
    """Train CIFAR-10 for a number of steps."""
    # tf.reset_default_graph()
    with tf.Graph().as_default():
      # print(FLAGS.stable)
      global_step = tf.contrib.framework.get_or_create_global_step()
      # Get images and labels for CIFAR-10.
      # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
      # GPU and resulting in a slow down.
      with tf.device('/cpu:0'):
        images, labels = cifar10.distorted_inputs()

      #print(parameters['w_conv1'])
      # Build a Graph that computes the logits predictions from the
      # inference model.
      if steps == 0:
        logits = cifar10.inference(images, keep_prob=1-FLAGS.dropout_prob, keep_prob2=1-FLAGS.dropout_prob2)
      else:
        logits = cifar10.inference(images, keep_prob=1-FLAGS.dropout_prob, keep_prob2=1-FLAGS.dropout_prob2, parameters=parameters)
      # Calculate loss.
      loss = cifar10.loss(logits, labels)

      # Build a Graph that trains the model with one batch of examples and
      # updates the model parameters.
      train_op = cifar10.train(loss, global_step, steps)

      class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def begin(self):
          self._step = -1
          self._start_time = time.time()
	  self._saver = tf.train.Saver()

	def after_create_session(self, session, coord):
	  if steps > 0:
            save_path = self._saver.save(session, "cifar10_dynamic_train_after_restruct/model.ckpt-{}".format(steps))
            with open("cifar10_dynamic_train_after_restruct/struct{}.txt".format(steps), "w") as file:
              file.write(str(parameters['b_conv1'].shape[0]) + ' ')
              file.write(str(parameters['b_conv2'].shape[0]) + ' ')
              file.write(str(parameters['w_fc1'].shape[1]) + ' ')
              file.write(str(parameters['w_fc2'].shape[1]))

        def before_run(self, run_context):
          self._step += 1
          return tf.train.SessionRunArgs(loss)  # Asks for loss value.

        def after_run(self, run_context, run_values):
          if self._step % FLAGS.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = float(duration / FLAGS.log_frequency)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), self._step + steps, loss_value,
                                 examples_per_sec, sec_per_batch))
          
	def end(self, session):
            # record variables
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	    cur_step = steps + FLAGS.max_steps
	    with open("steps.txt", "a") as file:
	      file.write(str(cur_step) + ' ')
	    if(cur_step >= 400000):
	      print("reaching end")
	      # FLAGS.stable = True
	      save_path = self._saver.save(session, "cifar10_dynamic_train/model.ckpt-{}".format(cur_step))
	      with open("cifar10_dynamic_train/struct.txt", "w") as file:
                file.write(str(parameters['b_conv1'].shape[0]) + ' ')
                file.write(str(parameters['b_conv2'].shape[0]) + ' ')
                file.write(str(parameters['w_fc1'].shape[1]) + ' ')
                file.write(str(parameters['w_fc2'].shape[1]))
	      return
            parameters['w_conv1'] = variables[1].eval(session=session)
            parameters['b_conv1'] = variables[2].eval(session=session)
            parameters['w_conv2'] = variables[3].eval(session=session)
            parameters['b_conv2'] = variables[4].eval(session=session)
            parameters['w_fc1'] = variables[5].eval(session=session)
            parameters['b_fc1'] = variables[6].eval(session=session)
            parameters['w_fc2'] = variables[7].eval(session=session)
            parameters['b_fc2'] = variables[8].eval(session=session)
            parameters['w_out'] = variables[9].eval(session=session)
            parameters['b_out'] = variables[10].eval(session=session)

	    save_path = self._saver.save(session, "cifar10_dynamic_train_before_restruct/model.ckpt-{}".format(cur_step))
            with open("cifar10_dynamic_train_before_restruct/struct{}.txt".format(cur_step), "w") as file:
              file.write(str(parameters['b_conv1'].shape[0]) + ' ')
              file.write(str(parameters['b_conv2'].shape[0]) + ' ')
              file.write(str(parameters['w_fc1'].shape[1]) + ' ')
              file.write(str(parameters['w_fc2'].shape[1]))

            # print(parameters['w_fc1'].shape)
            # print(parameters['b_fc1'].shape)
            # print(parameters['w_fc2'].shape)
	    dim1 = parameters['w_fc1'].shape[0]
	    dim2 = parameters['w_fc1'].shape[1]
	    dim3 = parameters['w_fc2'].shape[1]
	    # dim2 = int(dim2 * 0.9)
            # print(dim2)
            # parameters['w_fc1'] = parameters['w_fc1'][:, 0:dim2]
	    # parameters['b_fc1'] = parameters['b_fc1'][0:dim2]
	    # parameters['w_fc2'] = parameters['w_fc2'][0:dim2, :]
            if not FLAGS.stable:
	      if dim2 == dim3:
	        print("reduce layer fc1")
                FLAGS.stable, selected, r = decomposition.random_dropout(parameters['w_fc2'], dim3, dim2)
		FLAGS.dropout_prob = min(FLAGS.dropout_prob * r / dim2, 1)
		dim2 = r
	        parameters['w_fc1'] = parameters['w_fc1'][:, selected]
                parameters['b_fc1'] = parameters['b_fc1'][selected]
                parameters['w_fc2'] = parameters['w_fc2'][selected, :]
	      else:
	        print("reduce layer fc2")
		FLAGS.stable, selected, r = decomposition.random_dropout(parameters['w_fc2'], dim3, dim2)
		FLAGS.dropout_prob2 = min(FLAGS.dropout_prob2 * r / dim3, 1)
		dim3 = r
	        parameters['w_fc2'] = parameters['w_fc2'][:, selected]
		parameters['b_fc2'] = parameters['b_fc2'][selected]
		parameters['w_out'] = parameters['w_out'][selected, :]
		FLAGS.max_steps *= 2
	      if FLAGS.stable:
		FLAGS.max_steps = max_training_steps - cur_step
	      print("reduced size info, with r={}".format(r))
              print(parameters['w_fc1'].shape)
              print(parameters['b_fc1'].shape)
              print(parameters['w_fc2'].shape)
              print(parameters['b_fc2'].shape)
              print(parameters['w_out'].shape)
              print(parameters['b_out'].shape)

      with tf.train.MonitoredTrainingSession(
          checkpoint_dir=FLAGS.train_dir,
          hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                 tf.train.NanTensorHook(loss),
                 _LoggerHook()],
	  save_checkpoint_secs=None,
          config=tf.ConfigProto(
              log_device_placement=FLAGS.log_device_placement)) as mon_sess:
        while not mon_sess.should_stop():
          mon_sess.run(train_op)
      steps += FLAGS.max_steps
      # print("steps: %d" % steps)
  

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  if tf.gfile.Exists("cifar10_dynamic_train_before_restruct"):
    tf.gfile.DeleteRecursively("cifar10_dynamic_train_before_restruct")
  tf.gfile.MakeDirs("cifar10_dynamic_train_before_restruct")
  if tf.gfile.Exists("cifar10_dynamic_train_after_restruct"):
    tf.gfile.DeleteRecursively("cifar10_dynamic_train_after_restruct")
  tf.gfile.MakeDirs("cifar10_dynamic_train_after_restruct")
  dynamic_train()


if __name__ == '__main__':
  tf.app.run()
