import re
import tensorflow as tf

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
        return deserialize_genome(f.read())

def write_genome(genome, fname):
    with open(fname, 'w') as f:
        f.write(serialize_genome(genome))

def deserialize_genome(serialized):
    genome = {'connections': {}}
    lines = serialized.split('\n')
    for line in lines:
        if line == "": continue
        nums = re.findall(r'-?\d+', line)
        nums = [int(n) for n in nums]
        if line.startswith("INPUT"):
            genome['input'] = nums[0]
        elif line.startswith("OUTPUT"):
            genome['output'] = nums[0]
        else:
            print(line, nums)
            if nums[1] not in genome['connections']:
                genome['connections'][nums[1]] = []
            genome['connections'][nums[1]].append(nums[0])
    return genome

def serialize_genome(genome):
    out = 'INPUT(%d)\n' % genome['input']
    conn_keys = get_ordered_keys(genome)
    for key in conn_keys:
        for inp in genome['connections'][key]:
            out += "%d->%d\n" % (inp, key)
    out += 'OUTPUT(%d)' % genome['output']
    return out

def get_ordered_keys(genome):
    conn_keys = genome['connections'].keys()
    conn_keys.sort()
    conn_keys = [k for k in conn_keys if k >= 0] + [k for k in conn_keys if k < 0]
    return conn_keys

def build_model(genome):
    with tf.compat.v1.name_scope('input'):
        x = tf.compat.v1.placeholder(tf.float32, [None, genome['input']], name='x-input')
        y_ = tf.compat.v1.placeholder(tf.int64, [None, genome['output']], name='y-input')
    nodes = []
    output_nodes = []
    for idx in range(genome['input']):
        node = tf.compat.v1.gather(x, [idx], axis=1)
        nodes.append(node)
    conn_keys = get_ordered_keys(genome)
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
    print("OUT", output_nodes)
    y = tf.concat(output_nodes, 1)
    return x, y, y_

if __name__ == "__main__":
    g = read_genome('genome.txt')
    write_genome(g, 'genome2.txt')

