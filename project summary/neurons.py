#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:23:42 2019

@author: xihajun
"""

# First let's import all the tools needed
# Some basic tools
import time, os, argparse, io
dir = os.path.dirname(os.path.realpath(__file__))

# Tensorflow and numpy!
import tensorflow as tf
import numpy as np

# Matplotlib, so we can graph our functions
# The Agg backend is here for those running this on a server without X sessions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Our UA function
def univAprox(x, hidden_dim=50):
    # The simple case is f: R -> R
    input_dim = 1 
    output_dim = 1

    with tf.variable_scope('UniversalApproximator'):
        ua_w = tf.get_variable(
            name='ua_w'
            , shape=[input_dim, hidden_dim]
            , initializer=tf.random_normal_initializer(stddev=.1)
        )
        ua_b = tf.get_variable(
            name='ua_b'
            , shape=[hidden_dim]
            , initializer=tf.constant_initializer(0.)
        )
        z = tf.matmul(x, ua_w) + ua_b
        a = tf.nn.relu(z) # we now have our hidden_dim activations


        ua_v = tf.get_variable(
            name='ua_v'
            , shape=[hidden_dim, output_dim]
            , initializer=tf.random_normal_initializer(stddev=.1)
        )
        z = tf.matmul(a, ua_v)

    return z

# We define the function we want to approximate
def func_to_approx(x):
    s1 = tf.sigmoid(-30*(x-22.5))
    s2 = tf.sigmoid(30*(x-22.5))
    y = (x+3)*s1+(x-23)*s2
    return(y)



if __name__ == '__main__': # When we call the script directly ...
    # ... we parse a potentiel --nb_neurons argument 
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_neurons", default=500, type=int, help="Number of neurons or the UA")
    args = parser.parse_args()

    # We build the computation graph
    with tf.variable_scope('Graph') as scope:
        # Our inputs will be a batch of values taken by our functions
        x = tf.placeholder(tf.float32, shape=[None, 1], name="x")

        # We define the ground truth and our approximation 
        y_true = func_to_approx(x)
        y = univAprox(x, args.nb_neurons)

        # We define the resulting loss and graph it using tensorboard
        with tf.variable_scope('Loss'):
            loss = tf.reduce_mean(tf.square(y - y_true))
            # (Note the "_t" suffix here. It is pretty handy to avoid mixing 
            # tensor summaries and their actual computed summaries)
            loss_summary_t = tf.summary.scalar('loss', loss) 

        # We define our train operation using the Adam optimizer
        adam = tf.train.AdamOptimizer(learning_rate=1e-2)
        train_op = adam.minimize(loss)

    # This is some tricks to push our matplotlib graph inside tensorboard
    with tf.variable_scope('TensorboardMatplotlibInput') as scope:
        # Matplotlib will give us the image as a string ...
        img_strbuf_plh = tf.placeholder(tf.string, shape=[]) 
        # ... encoded in the PNG format ...
        my_img = tf.image.decode_png(img_strbuf_plh, 4) 
        # ... that we transform into an image summary
        img_summary = tf.summary.image( 
            'matplotlib_graph'
            , tf.expand_dims(my_img, 0)
        ) 

    # We create a Saver as we want to save our UA after training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # We create a SummaryWriter to save data for TensorBoard
        result_folder = dir + '/results/' + str(int(time.time()))
        sw = tf.summary.FileWriter(result_folder, sess.graph)

        print('Training our universal approximator')
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            # We uniformly select a lot of points for a good approximation ...
            x_in = np.random.uniform(0, 26, [100000, 1])
            # ... and train on it
            current_loss, loss_summary, _ = sess.run([loss, loss_summary_t, train_op], feed_dict={
                x: x_in
            })
            # We leverage tensorboard by keeping track of the loss in real time
            sw.add_summary(loss_summary, i + 1)

            if (i + 1) % 100 == 0:
                print('batch: %d, loss: %f' % (i + 1, current_loss))

        print('Plotting graphs')
        # We compute a dense enough graph of our functions
        inputs = np.vstack((np.array([[i/100] for i in range(2200)]),np.array([[i/100] for i in range(2300,2600)])))
        y_true_res, y_res = sess.run([y_true, y], feed_dict={
            x: inputs
        })
        # We plot it using matplotlib
        # (This is some matplotlib wizardry to get an image as a string,
        # read the matplotlib documentation for more information)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(inputs, y_true_res.flatten())
        plt.subplot(212)
        plt.plot(inputs, y_res)
        imgdata = io.BytesIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        # We push our graph into TensorBoard
        plot_img_summary = sess.run(img_summary, feed_dict={
            img_strbuf_plh: imgdata.getvalue()
        })
        sw.add_summary(plot_img_summary, i + 1)
        plt.clf()

        # Finally we save the graph to check that it looks like what we wanted
        saver.save(sess, result_folder + '/data.chkp')