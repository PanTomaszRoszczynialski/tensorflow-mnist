import pickle
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_mnist():
    """ Prepare mnist data """
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

def make_softmax(nof_steps = 100):
    """ Official example turned into function """
    mnist = get_mnist()

    # Declare variables
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]), name = 'W')
    b = tf.Variable(tf.zeros([10]), name = 'b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Training concepts
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    # Train with batches
    for i in range(nof_steps):
        batch_xs, batch_ys = mnist.train.next_batch(500)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Try to save the results
    save_path = saver.save(sess, 'tmp/yo.ckpt')

    # This clears TF variable names and allows saving and loading
    # withing one python session
    tf.reset_default_graph()

    print 'something saved in:', save_path, 'with {} steps'.format(nof_steps)

def test_softmax():
    """ Load model from disk """
    mnist = get_mnist()

    # Declare variables
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]), name = 'W')
    b = tf.Variable(tf.zeros([10]), name = 'b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Training concepts
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    saver = tf.train.Saver()

    sess = tf.Session()

    saver.restore(sess, 'tmp/yo.ckpt')

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # This clears TF variable names and allows saving and loading
    # withing one python session
    tf.reset_default_graph()

    score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print 'achieved accuracy: ', score

    return score

if __name__ == '__main__':
    """ This isi sihT """
    scores = []
    steps = [100 * it for it in range(1, 200)]

    for howmany in steps:
        make_softmax(howmany)
        scores.append(test_softmax())

        with open('scores.pickle', 'wb') as fin:
              pickle.dump(scores, fin)
