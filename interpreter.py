import re
import tensorflow as tf
from random import randint

MAIN_DATA = [
  {'x': [0.0, 0.0], 'y': [0.0]},
  {'x': [1.0, 1.0], 'y': [0.0]},
  {'x': [1.0, 0.0], 'y': [1.0]},
  {'x': [0.0, 1.0], 'y': [1.0]},
]

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)

def read_genome(fname):
    genome = {'connections': {}}
    with open(fname) as f:
        for line in f.readlines():
            nums = re.findall(r'-?\d+', line)
            nums = [int(n) for n in nums]
            if line.startswith("INPUT"):
                genome['input'] = nums[0]
            elif line.startswith("OUTPUT"):
                genome['output'] = nums[0]
            else:
                if nums[1] not in genome['connections']:
                    genome['connections'][nums[1]] = []
                genome['connections'][nums[1]].append(nums[0])
    return genome

def build_model(genome):
    with tf.compat.v1.name_scope('input'):
        x = tf.compat.v1.placeholder(tf.float32, [None, genome['input']], name='x-input')
        y_ = tf.compat.v1.placeholder(tf.int64, [None, genome['output']], name='y-input')
    nodes = []
    output_nodes = []
    for idx in range(genome['input']):
        node = tf.compat.v1.gather(x, [idx], axis=1)
        nodes.append(node)
    conn_keys = genome['connections'].keys()
    conn_keys.sort()
    conn_keys = [k for k in conn_keys if k >= 0] + [k for k in conn_keys if k < 0]
    for conn in conn_keys:
        shape = [len(genome['connections'][conn]), 1]
        with tf.compat.v1.name_scope('weights'):
            weights = tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))
            variable_summaries(weights)
        with tf.compat.v1.name_scope('biases'):
            bias = tf.Variable(tf.constant(0.1, shape=[1]))
            variable_summaries(bias)
        input_tensors = [nodes[up] for up in genome['connections'][conn]]
        input = tf.concat(input_tensors, 1)
        node = tf.matmul(input, weights) + bias
        node = tf.nn.relu(node)
        if conn < 0:
            output_nodes.append(node)
        else:
            nodes.append(node)
    y = tf.concat(output_nodes, 1)
    return x, y, y_

if __name__ == '__main__':
    sess = tf.compat.v1.InteractiveSession()
    x, y, y_ = build_model(read_genome('genome.txt'))
    tf.compat.v1.global_variables_initializer().run()

    with tf.compat.v1.name_scope('absolute_difference'):
        with tf.compat.v1.name_scope('total'):
            absolute_difference = tf.compat.v1.losses.absolute_difference(
                    labels=y_, predictions=y)
    tf.compat.v1.summary.scalar('absolute_difference', absolute_difference)

    with tf.compat.v1.name_scope('train'):
        train_step = tf.compat.v1.train.AdamOptimizer(0.001).minimize(absolute_difference)

    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter('log/train',
                                                                                                 sess.graph)
    test_writer = tf.compat.v1.summary.FileWriter('log/test')
    tf.compat.v1.global_variables_initializer().run()

    def feed_dict(train):
        xs = []
        ys = []
        for i in range(100):
            item = MAIN_DATA[randint(0, 3)]
            xs.append(item['x'])
            ys.append(item['y'])
        return {x: xs, y_: ys}

    for i in range(10000):
        if i % 10 == 0:    # Record summaries and test-set absolute_difference
            summary, acc = sess.run([merged, absolute_difference], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:    # Record train set summaries, and train
            if i % 100 == 99:    # Record execution stats
                run_options = tf.compat.v1.RunOptions(
                        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                    feed_dict=feed_dict(True),
                                    options=run_options,
                                    run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:    # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()

