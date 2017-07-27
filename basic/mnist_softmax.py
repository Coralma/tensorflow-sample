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

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # 导入数据
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # 创建数据模型，placeholder表示输入数据的地方（None为不限定输入条数，784表示每条输入是一个784维的向量）
  # *之所以使用784是因为图片矩阵为28*28，每个像素为一个矩阵点。
  x = tf.placeholder(tf.float32, [None, 784])
  # W 表示weights权重，同样784维数，10个分类
  W = tf.Variable(tf.zeros([784, 10]))
  # b 表示biases,数据倾向性指标
  b = tf.Variable(tf.zeros([10]))
  # 实现Softmax Regression算法
  y = tf.matmul(x, W) + b

  # 为了训练模型，需要定loss function
  y_ = tf.placeholder(tf.float32, [None, 10])

  # 通常采用cross_entropy作为
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # 创建session,并把当前session注册为默认session
  sess = tf.InteractiveSession()
  # 初始化变量并执行run方法
  tf.global_variables_initializer().run()
  # 迭代执行训练操作
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # 测试验证训练模型
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)