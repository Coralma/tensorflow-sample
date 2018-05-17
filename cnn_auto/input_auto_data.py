import tensorflow as tf
import numpy as np
import os

def get_image(file_dir):
    benz = []
    toyota = []
    label_benz = []
    label_toyota = []
    ad = { 'Benz_GLA':1, 'Honda_CR-V':2, 'Toyota_Camry':3 } #{ Benz_GLA:1, }
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'Benz_GLA':
            benz.append(file_dir + file)
            label_benz.append(0)
        else:
            toyota.append(file_dir + file)
            label_toyota.append(1)

    image_list = np.hstack((benz, toyota))
    label_list = np.hstack((label_benz, label_toyota))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

train_dir = 'D:/projects-python/ai-training-data/data/autos/'
image_list, label_list = get_image(train_dir)
print("tested")