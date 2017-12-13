# 计算线性代数
import tensorflow as tf
import numpy as np #科学计算模块

# create data， X数据是100
x_data = np.random.rand(100).astype(np.float32)     # 生成100个随机数列
y_data = x_data*0.5 + 0.8  #线性函数(二元一次方程) 其中0.1为weight， 0.3为biases

# 创建结构
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # tf.Variable定义变量， 生成一个从-1到1的随机数
biases = tf.Variable(tf.zeros([1])) # 定义偏值为0
y = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data)) # 计算 y 和 y_data 的误差:
optimizer = tf.train.GradientDescentOptimizer(0.1) # 使用优化器减少误差,梯度下降优化器
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # 初始化变量
sess = tf.Session()
sess.run(init)          # Very important

for step in range(2000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

#print(np.random.rand(100).astype(np.float32))
#print(x_data)
#print(y_data)
print(Weights)