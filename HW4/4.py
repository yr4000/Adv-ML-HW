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
OUTPUT_SIZE = 1 #for lunar it should be size 4
TOTAL_EPISODES = 100
PERIOD = 10

ENVIRONMENT = cart_pole_env
env = gym.make(cart_pole_env)
env.reset()



#states - a 4XT matrix, holds all the states from up to the T step
       #notice it can be a matrix or a vector
#y = agent(observations)     #TODO: right now y = 0 becaue that's what the function returns... do i really want to do it in a function? i can tale the rest of the code after the y out.

# Defining the agent
#placeholder for the input
observations = tf.placeholder(tf.float32, [None, INPUT_SIZE])
# first layer:
# define weights 8xHIDDEN_NEURONS_NO
W1 = tf.Variable(tf.zeros([INPUT_SIZE, HIDDEN_NEURONS_NO]))
# define bias
b1 = tf.Variable(tf.zeros([HIDDEN_NEURONS_NO]))
# tanh them with observation
h1 = tf.nn.tanh(tf.matmul(observations,W1) + b1)

# second layer:
W2 = tf.Variable(tf.zeros([HIDDEN_NEURONS_NO, HIDDEN_NEURONS_NO]))
b2 = tf.Variable(tf.zeros([HIDDEN_NEURONS_NO]))
h2 = tf.nn.tanh(tf.matmul(h1,W2) + b2)

# third layer (output layer):
W3 = tf.Variable(tf.zeros([HIDDEN_NEURONS_NO, OUTPUT_SIZE]))
b3 = tf.Variable(tf.zeros([OUTPUT_SIZE]))
y = tf.nn.softmax(tf.matmul(h2,W3) + b3)

tvars = tf.trainable_variables()

# rewards sums from k=t to T:
rewards_arr = tf.placeholder(tf.float32, [None, 1])
# actions - a mask matrix which filters ys result accordint to the actions that were chosen. see boolean mask here: https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops/slicing_and_joining#boolean_mask
actions_mask = tf.placeholder(tf.bool, [None, OUTPUT_SIZE])
filtered_actions = tf.boolean_mask(y, actions_mask)  # should return a T size vector with correct (chosen) action values
pi = tf.log(filtered_actions)
#grad = tf.reduce_sum(tf.scalar_mul(1 // tf.size(pi), tf.multiply(pi,rewards_arr)))
grad_step = tf.reduce_sum(tf.multiply(pi,rewards_arr))
#Gradients = tf.gradients(grad_step,tvars)
# then train the network - for each of the parameters do the GD as described in the HW.
train_step = tf.train.AdamOptimizer(1e-2).apply_gradients(zip([grad_step for i in range(6)],tvars))

#TODO: answer the following questions:
'''
1) How exactly do i get the correct gradient? anf how do i use apply_gradients with them?
2) maybe i should run this guys code and see if i can learn something about the functions nature
3) I really should understand how exactly the adam optimiser works...
4) compare between how you took the gradient step to his (what is reduce_mean exactly?)
5) need to understand also how exactly tf.gradients works, and if this corelates with what they asked.
6) how to i use scalar_mul?
'''


init = tf.global_variables_initializer()
def main(argv):
    print(tvars)
    rewards, states, actions_booleans = [], [], []
    episode_number, steps, total_reward = 0,0,0
    with tf.Session() as sess:
        sess.run(init)
        obsrv = env.reset() # Obtain an initial observation of the environment
        while episode_number <= TOTAL_EPISODES:
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(y,feed_dict={observations: obsrv})
            actions_booleans.append(np.random.multinomial(1, action_probs))
            if(ENVIRONMENT == cart_pole_env):
                action = 0 if action_probs < 0.5 else 1
            else:
                action = np.argmax(actions_booleans[-1])
            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)
            #add reward to rewards and obsrv to states
            rewards.append(reward)
            states.append(obsrv)
            steps += 1
            if done:
                #create the rewards sums array and reverse again
                rewards_sum = np.cumsum(rewards[::-1]);
                #normalize prizes and reverse
                rewards_sum = np.divide(rewards_sum[::-1],np.sum(rewards_sum))
                #modify actions_booleans to be an array of booleans
                actions_booleans = np.array(actions_booleans)
                actions_booleans = actions_booleans == 1
                #take the train step
                sess.run(train_step, feed_dict={observations: states,actions_mask:actions_booleans,rewards_arr: rewards_sum })
                total_reward += sum(rewards)
                if(episode_number%PERIOD==0):
                    # save with pickle
                    pickle.dump(tvars, open("tvars.pkl", "wb"))
                    print ('Average reward for episode %f.  Total average reward %f.' % (sum(rewards) / steps, total_reward / PERIOD))
                    total_reward = 0
                episode_number += 1
                steps = 0
                rewards, states, actions_booleans = [], [], []
                obsrv = env.reset()


if __name__ == '__main__':
    tf.app.run()
