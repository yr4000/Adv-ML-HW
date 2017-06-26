#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:02:28 2017

@author: daniel
"""
import pickle as pickle
import numpy as np
import tensorflow as tf
import gym
envd = 'LunarLander-v2'
env = gym.make(envd)

fd = 'n-bws.p'
f = open(fd,"rb")
ws = pickle.load(f)
D = 8

tf.reset_default_graph()

W1,b1,W2,b2,W3,b3 = tf.constant(ws[0]),tf.constant(ws[1]),tf.constant(ws[2]),tf.constant(ws[3]),tf.constant(ws[4]),tf.constant(ws[5])
observations = tf.placeholder(tf.float32, [1,D] , name="input_x")
def agent(ob):
    layer1 = tf.nn.tanh(tf.matmul(ob,W1)+b1)
    layer2 = tf.nn.tanh(tf.matmul(layer1,W2)+b2)
    score = tf.matmul(layer2,W3)+b3
    probability = tf.nn.softmax(score)
    return probability

probability = agent(observations)

total_eps = 20000
ep_num = 0
cum_rwrd = 0.
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    observation = env.reset()
    episode_num = 0
    reward_per_ep = 0
    while ep_num<total_eps:
        #env.render()
        x = np.reshape(observation,[1,D])
        tfprob = sess.run(probability,feed_dict={observations: x})
        
        # you might need this for numerical stability:
        tfprob = np.float64(tfprob)
        tfprob = tfprob/sum(tfprob[0])
        
        action = np.argmax(np.random.multinomial(1,tfprob[0],1)[0])
        observation, reward, done, info = env.step(action)
        reward_per_ep += reward
        cum_rwrd += reward
        if done:
            avg = cum_rwrd/(ep_num+1)
            print ('ep:',ep_num,'rwrd:',reward_per_ep,'avg:',avg)
            reward_per_ep=0
            ep_num+=1
            observation = env.reset()
