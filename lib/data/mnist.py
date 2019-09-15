from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

image_size = 28
labels_size = 10
learning_rate = 0.05
steps_number = 1000
batch_size = 100

def feed_dict(x, y_, train):
    if train:
        xs, ys = mnist.train.next_batch(batch_size)
    else:
        xs, ys = mnist.test.next_batch(batch_size)
    return {x: xs, y_: ys}

