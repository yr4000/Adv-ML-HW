#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:11:02 2017

@author: daniel
"""
from HW4 import gym
import numpy as np
import pickle as pickle
import tensorflow as tf

env_d = 'LunarLander-v2'
cart_pole_env = 'CartPole-v0'
INPUT_SIZE = 4 #for lunar it should be size 8
HIDDEN_NEURONS_NO = 15
OUTPUT_SIZE = 1 #for lunar it should be size 4
TOTAL_EPISODES = 0

env = gym.make(cart_pole_env)
env.reset()

def agent(observation):     #TODO: this should be the NN?
    # Defining the layers:
    # first layer:
    # define weights 8xHIDDEN_NEURONS_NO
    W1 = tf.Variable(tf.zeros([INPUT_SIZE, HIDDEN_NEURONS_NO]))
    # define bias
    b1 = tf.Variable(tf.zeros([HIDDEN_NEURONS_NO]))
    # tanh them with observation
    h1 = tf.nn.tanh(tf.matmul(W1, observation) + b1)

    # second layer:
    W2 = tf.Variable(tf.zeros([HIDDEN_NEURONS_NO, HIDDEN_NEURONS_NO]))
    b2 = tf.Variable(tf.zeros([HIDDEN_NEURONS_NO]))
    h2 = tf.nn.tanh(tf.matmul(W2, h1) + b2)

    # third layer (output layer):
    W3 = tf.Variable(tf.zeros([HIDDEN_NEURONS_NO, OUTPUT_SIZE]))
    b3 = tf.Variable(tf.zeros([OUTPUT_SIZE]))
    y = tf.nn.softmax(tf.matmul(W3, h2) + b3)

    # rewards sums from k=t to T:
    rewards_sums = tf.placeholder(tf.float32, [None,1])
    #actions - a mask matrix which filters ys result accordint to the actions that were chosen. see boolean mask here: https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops/slicing_and_joining#boolean_mask
    actions = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
    filtered_actions = tf.boolean_mask(y,actions)       #should return a T size vector with correct (chosen) action values
    pi = tf.log(filtered_actions)
    grad = tf.scalar_mul(1/tf.size(pi),tf.reduce_sum(pi))
    # then train the network - for each of the parameters do the GD as described in the HW.
    train_step = tf.train.AdamOptimizer(1e-2).apply_gradients(grad)

    return 0

#states - a 4XT matrix, holds all the states from up to the T step
observations = tf.placeholder(tf.float32, [None, INPUT_SIZE])       #notice it can be a matrix or a vector
y = agent(observations)     #TODO: right now y = 0 becaue that's what the function returns... do i really want to do it in a function? i can tale the rest of the code after the y out.

init = tf.global_variables_initializer()
def main(argv):
    rewards, states = [], []
    episode_number = 0
    with tf.Session() as sess:
        sess.run(init)
        obsrv = env.reset() # Obtain an initial observation of the environment
        while episode_number <= TOTAL_EPISODES:
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(y,feed_dict={observations: obsrv})
            action = np.argmax(np.multinomial(1, action_probs))     #TODO: figure out exactly what multinomial does
            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)
            #TODO: complete the following shit:
            #add reward to rewards and obsrv to states
            rewards.append(reward)
            states.append(obsrv)
            #TODO: create the mask actions matrix! should be TX4 size
            if done:
                #TODO: create the rewards sums array AND REVERSE IT!
                #TODO: normalize prizes
                #TODO: take the train step
                #TODO: save with pickle
                episode_number += 1
                obsrv = env.reset()


if __name__ == '__main__':
    tf.app.run()
