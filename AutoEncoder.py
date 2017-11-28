import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

folder_path = 'sample-png'

def Load_Image(filename):
    img = mpimg.imread(filename)
    # imgplot = plt.imshow(img[:,:,:]); plt.show()
    return img

def Load_Image_List(folder):
    print('Load Images from Folder: "%s"' % (folder))
    img_list = []
    file_list = os.listdir(folder)
    for file_name in file_list:
        file_path = os.path.join(folder, file_name)
        img_list.append(Load_Image(file_path))
    # print(len(img_list))
    return np.asarray(img_list)


images = Load_Image_List(folder_path)
n = len(images)

def Next_Batch(batch_size):
    batch_indices = np.random.permutation(n)[:batch_size]
    return images[batch_indices]


def Test_AutoEncoder():
    import tensorflow as tf
    def Init_Var(shape, stddev=0.1):
        return tf.truncated_normal(shape=shape, stddev=stddev)

    def Init_Bias(shape, stddev=0.0):
        return tf.truncated_normal(shape=shape, stddev=stddev)

    img_shape = [400, 400, 3]
    filter_shape = [5, 5]
    x = tf.placeholder(tf.float32, [None] + img_shape)
    learning_rate = tf.placeholder(tf.float32)

    # ========== ENCODER ==========#
    e1_w = tf.Variable(Init_Var(shape=filter_shape + [3, 16]))
    e1_b = tf.Variable(Init_Bias(shape=[200, 200, 16]))
    e1_s = tf.nn.relu(tf.nn.convolution(x, e1_w, strides=[2, 2], padding='SAME') + e1_b)
    print(e1_s)

    e2_w = tf.Variable(Init_Var(shape=filter_shape + [16, 8]))
    e2_b = tf.Variable(Init_Bias(shape=[100, 100, 8]))
    e2_s = tf.nn.relu(tf.nn.convolution(e1_s, e2_w, strides=[2, 2], padding='SAME') + e2_b)
    print(e2_s)

    e3_w = tf.Variable(Init_Var(shape=filter_shape + [8, 4]))
    e3_b = tf.Variable(Init_Bias(shape=[50, 50, 4]))
    e3_s = tf.nn.relu(tf.nn.convolution(e2_s, e3_w, strides=[2, 2], padding='SAME') + e3_b)
    print(e3_s)

    e4_w = tf.Variable(Init_Var(shape=filter_shape + [4, 2]))
    e4_b = tf.Variable(Init_Bias(shape=[25, 25, 2]))
    e4_s = tf.nn.relu(tf.nn.convolution(e3_s, e4_w, strides=[2, 2], padding='SAME') + e4_b)
    print(e4_s)

    # ========== DECODER ==========#
    resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    d4_up = tf.image.resize_images(e4_s, size=(50, 50), method=resize_method)
    d4_w = tf.Variable(Init_Var(shape=filter_shape + [2, 4]))
    d4_b = tf.Variable(Init_Bias(shape=[50, 50, 4]))
    d4_s = tf.nn.relu(tf.nn.convolution(d4_up, d4_w, strides=[1, 1], padding='SAME') + d4_b)
    print(d4_s)

    d3_up = tf.image.resize_images(d4_s, size=(100, 100), method=resize_method)
    d3_w = tf.Variable(Init_Var(shape=filter_shape + [4, 8]))
    d3_b = tf.Variable(Init_Bias(shape=[100, 100, 8]))
    d3_s = tf.nn.relu(tf.nn.convolution(d3_up, d3_w, strides=[1, 1], padding='SAME') + d3_b)
    print(d3_s)

    d2_up = tf.image.resize_images(d3_s, size=(200, 200), method=resize_method)
    d2_w = tf.Variable(Init_Var(shape=filter_shape + [8, 16]))
    d2_b = tf.Variable(Init_Bias(shape=[200, 200, 16]))
    d2_s = tf.nn.relu(tf.nn.convolution(d2_up, d2_w, strides=[1, 1], padding='SAME') + d2_b)
    print(d2_s)

    d1_up = tf.image.resize_images(d2_s, size=(400, 400), method=resize_method)
    d1_w = tf.Variable(Init_Var(shape=filter_shape + [16, 3]))
    d1_b = tf.Variable(Init_Bias(shape=[400, 400, 3]))
    d1_s = tf.nn.convolution(d1_up, d1_w, strides=[1, 1], padding='SAME') + d1_b
    print(d1_s)

    # reconstructed_imgs = tf.clip_by_value(d1_s, 0.0, 1.0)
    reconstructed_imgs = tf.nn.sigmoid(d1_s)
    cost1 = tf.losses.mean_squared_error(x, reconstructed_imgs)
    cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=d1_s))
    cost = cost2# + cost2
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # ========== IMPLEMENTATION of GRAPH ==========#
    print('#========== IMPLEMENTATION OF TRAINING ==========#')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch_size = 4
    lr = 0.0001
    for epoch in range(2000):
        img_batch = Next_Batch(batch_size)
        _train, train_cost = sess.run([opt, cost], feed_dict={x: img_batch, learning_rate: lr})
        if (epoch + 1) % 100 == 0: print(' Epoch %d: %.6f' % (epoch + 1, train_cost))
        if (epoch + 1) % 10 == 0 and epoch < 1000: lr *= 0.8

    test_batch = Next_Batch(1)
    _img = sess.run(reconstructed_imgs, feed_dict={x: test_batch})[0]
    print(_img.shape)
    plt.imshow(_img); plt.show()
    plt.imshow(test_batch[0]); plt.show()


if __name__ == '__main__':
    Test_AutoEncoder()