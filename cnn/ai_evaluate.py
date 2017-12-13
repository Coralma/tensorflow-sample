from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import input_data
import model
import numpy as np
import os

def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]
   image = Image.open(img_dir)
   plt.imshow(image)
   plt.show()
   image = image.resize([208, 208])
   image = np.array(image)
   return image

def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''
   # you need to change the directories to yours.
   #train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
   test_dir = 'D:/projects-python/ai-training-data/data/test/'
   logs_train_dir = 'D:/projects-python/ai-training-data/logs/train/'

   train, train_label = input_data.get_files(test_dir)
   image_array = get_one_image(train)
   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2
       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 208, 208, 3])
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)
       logit = tf.nn.softmax(logit)
       x = tf.placeholder(tf.float32, shape=[208, 208, 3])
       # you need to change the directories to yours.
       #logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'
       saver = tf.train.Saver()
       with tf.Session() as sess:
           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')
           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a cat')
           else:
               print('This is a dog')
           print('The possibility of cat is %.6f' %prediction[:, 0])
           print('The possibility of dog is %.6f' %prediction[:, 1])

def get_test_image(test_dir):
    print(test_dir)
    imgs = []
    for file in os.listdir(test_dir):
        imgs.append(test_dir + file)
    return imgs

data = get_test_image('D:/projects-python/ai-training-data/data/test/')
print(data)
#evaluate_one_image()