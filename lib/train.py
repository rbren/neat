import tensorflow as tf

from evolver import evolve
from interpreter import build_model, read_genome
from data import feed_dict

STEPS_PER_GENERATION = 2500

def train(genome):
    sess = tf.compat.v1.InteractiveSession()
    x, y, y_ = build_model(genome)
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

    for i in range(STEPS_PER_GENERATION):
        if i % int(STEPS_PER_GENERATION / 5) == 0:    # Record summaries and test-set absolute_difference
            summary, loss = sess.run([merged, absolute_difference], feed_dict=feed_dict(x, y_, False))
            test_writer.add_summary(summary, i)
            print('Loss at step %s: %s' % (i, loss))
        else:    # Record train set summaries, and train
            if i % 100 == 99:    # Record execution stats
                run_options = tf.compat.v1.RunOptions(
                        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                    feed_dict=feed_dict(x, y_, True),
                                    options=run_options,
                                    run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
            else:    # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(x, y_, True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
    sess.close()
    tf.reset_default_graph()
    return loss

if __name__ == '__main__':
    genome = read_genome('genome.txt')
    loss = 1000

    for i in range(100):
        new_genome = evolve(genome)
        print(genome)
        print(new_genome)
        new_loss = train(new_genome)
        print("loss", loss, new_loss)
        if new_loss < loss:
            print("IMPROVE!")
            genome = new_genome
            loss = new_loss
