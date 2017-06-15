#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:11:02 2017

@author: daniel
"""
import gym
import numpy as np
import pickle as pickle
import tensorflow as tf

env_d = 'LunarLander-v2'
cart_pole_env = 'CartPole-v0'
INPUT_SIZE = 4 #for lunar it should be size 8
HIDDEN_NEURONS_NO = 15
OUTPUT_SIZE = 2 #for lunar it should be size 4
TOTAL_EPISODES = 10000
PERIOD = 100
DEFINE_COPY = True      #TODO: change the name...

ENVIRONMENT = cart_pole_env
env = gym.make(ENVIRONMENT)
env.reset()



#states - a 4XT matrix, holds all the states from up to the T step
       #notice it can be a matrix or a vector
#y = agent(observations)     #TODO: right now y = 0 becaue that's what the function returns... do i really want to do it in a function? i can tale the rest of the code after the y out.

# Defining the agent
# placeholder for the input
observations = tf.placeholder(tf.float32, [None, INPUT_SIZE])

if(DEFINE_COPY):
    # first layer:
    # define weights 8xHIDDEN_NEURONS_NO
    W1 = tf.get_variable(name="W1", shape=[INPUT_SIZE, HIDDEN_NEURONS_NO],
                         initializer=tf.contrib.layers.xavier_initializer())
    # define bias
    b1 = tf.get_variable(name="b1", shape=[HIDDEN_NEURONS_NO], initializer=tf.contrib.layers.xavier_initializer())

    # second layer:
    W2 = tf.get_variable(name="W2", shape=[HIDDEN_NEURONS_NO, HIDDEN_NEURONS_NO],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name="b2", shape=[HIDDEN_NEURONS_NO], initializer=tf.contrib.layers.xavier_initializer())

    # third layer (output layer):
    W3 = tf.get_variable(name="W3", shape=[HIDDEN_NEURONS_NO, OUTPUT_SIZE],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(name="b3", shape=[OUTPUT_SIZE], initializer=tf.contrib.layers.xavier_initializer())

else:
    # Defining the agent
    #placeholder for the input
    observations = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    # first layer:
    # define weights 8xHIDDEN_NEURONS_NO
    W1 = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_NEURONS_NO]))
    # define bias
    b1 = tf.Variable(tf.random_normal([HIDDEN_NEURONS_NO]))

    # second layer:
    W2 = tf.Variable(tf.random_normal([HIDDEN_NEURONS_NO, HIDDEN_NEURONS_NO]))
    b2 = tf.Variable(tf.random_normal([HIDDEN_NEURONS_NO]))

    # third layer (output layer):
    W3 = tf.Variable(tf.random_normal([HIDDEN_NEURONS_NO, OUTPUT_SIZE]))
    b3 = tf.Variable(tf.random_normal([OUTPUT_SIZE]))

#first,second and third layer computations
h1 = tf.nn.tanh(tf.matmul(observations, W1) + b1)
h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
y = tf.nn.softmax(tf.matmul(h2, W3) + b3)  # TODO: there is some problem with the softmax

tvars = tf.trainable_variables()

# rewards sums from k=t to T:
rewards_arr = tf.placeholder(tf.float32, [1,None])
# actions - a mask matrix which filters ys result accordint to the actions that were chosen. see boolean mask here: https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops/slicing_and_joining#boolean_mask
actions_mask = tf.placeholder(tf.bool, [None, OUTPUT_SIZE])
filtered_actions = tf.boolean_mask(y, actions_mask)  # should return a T size vector with correct (chosen) action values
pi = tf.log(filtered_actions)       #TODO: log(0) = - inf!!!!!
#grad = tf.reduce_sum(tf.scalar_mul(1 // tf.size(pi), tf.multiply(pi,rewards_arr)))
grad_step = tf.divide(tf.reduce_sum(tf.multiply(pi,rewards_arr)),tf.to_float(tf.size(pi)))
Gradients = tf.gradients(grad_step,tvars)
# then train the network - for each of the parameters do the GD as described in the HW.
train_step = tf.train.AdamOptimizer(1e-2).apply_gradients(zip(Gradients,tvars))
#train_step = tf.train.AdamOptimizer(1e-2).minimize(grad_step)

#TODO: answer the following questions:
'''
1) How exactly do i get the correct gradient? and how do i use apply_gradients with them?
2) maybe i should run this guys code and see if i can learn something about the functions nature
3) I really should understand how exactly the adam optimiser works...
4) compare between how you took the gradient step to his (what is reduce_mean exactly?)
5) need to understand also how exactly tf.gradients works, and if this corelates with what they asked.
new:
6) maybe i need to zero somethings like he does every once in a while?
'''


init = tf.global_variables_initializer()
def main(argv):
    rewards, states, actions_booleans = [], [], []
    episode_number, steps, total_reward = 0,0,0
    action_sum = np.array([[0.0,0.0]])      #TODO: for debug
    action_probs_arr = []
    with tf.Session() as sess:
        sess.run(init)
        obsrv = env.reset() # Obtain an initial observation of the environment
        while episode_number <= TOTAL_EPISODES:
            #append relevant observation to action to states
            states.append(obsrv)
            modified_obsrv = np.reshape(obsrv, [1, INPUT_SIZE])
            '''
            temp = sess.run(tvars)      #TODO: for debug
            th1 = sess.run(h1,feed_dict={observations: modified_obsrv})
            th2 = sess.run(h2,feed_dict={observations: modified_obsrv})
            th3 = sess.run(y,feed_dict={observations: modified_obsrv})
            '''
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(y,feed_dict={observations: modified_obsrv})
            action_probs_arr.append(action_probs)
            actions_booleans.append(np.random.multinomial(1, action_probs[0]))
            '''
            if(ENVIRONMENT == cart_pole_env):
                action = 0 if action_probs[[0]] < 0.5 else 1
            else:
            '''
            action = np.argmax(actions_booleans[-1])
            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)
            #add reward to rewards and obsrv to states
            rewards.append(reward)
            steps += 1
            action_sum += np.array(action_probs)
            if done:
                #create the rewards sums array and reverse again
                rewards_sums = np.cumsum(rewards[::-1]);
                #normalize prizes and reverse
                rewards_sums = np.divide(rewards_sums[::-1],np.sum(rewards_sums))
                modified_rewards_sums = np.reshape(rewards_sums, [1, len(rewards_sums)])
                #modify actions_booleans to be an array of booleans
                debug_ab = actions_booleans
                actions_booleans = np.array(actions_booleans)
                actions_booleans = actions_booleans == 1
                #gradients for check.
                # TODO: the unused are for debug
                '''
                th3 = sess.run(y, feed_dict={observations: states})
                fa = sess.run(filtered_actions, feed_dict={observations: states,actions_mask:actions_booleans,rewards_arr: modified_rewards_sums })
                pi_debug = sess.run(pi, feed_dict={observations: states,actions_mask:actions_booleans,rewards_arr: modified_rewards_sums })
                loss_debug = sess.run(grad_step, feed_dict={observations: states,actions_mask:actions_booleans,rewards_arr: modified_rewards_sums })
                grads = sess.run(Gradients, feed_dict={observations: states,actions_mask:actions_booleans,rewards_arr: modified_rewards_sums })
                '''
                #take the train step
                sess.run(train_step, feed_dict={observations: states,actions_mask:actions_booleans,rewards_arr: modified_rewards_sums })
                total_reward += sum(rewards)
                if(episode_number%PERIOD==0):
                    print ('Average reward for episode %f.  Total average reward %f.' % (episode_number, total_reward / PERIOD))
                    #print("actions_sum = ", action_sum)
                    total_reward = 0
                episode_number += 1
                steps = 0
                rewards, states, actions_booleans = [], [], []
                action_probs_arr = [] #TODO: for debug
                obsrv = env.reset()


if __name__ == '__main__':
    tf.app.run()
    # save with pickle
    #TODO: doesn't work...
    #f = open("tvars.pkl", "wb")
    #pickle.dump(tvars, f)
    #f.close()
