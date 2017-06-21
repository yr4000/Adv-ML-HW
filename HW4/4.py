#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:11:02 2017

@author: daniel
"""
import gym
import numpy as np
import pickle as pkl
import tensorflow as tf
import matplotlib
matplotlib.use('Qt4Agg')
#matplotlib.use('Agg') #in order not to display the plot
import matplotlib.pyplot as plt
import os


env_d = 'LunarLander-v2'
cart_pole_env = 'CartPole-v0'
INPUT_SIZE = 8 #for lunar it should be size 8, for cartPole 4
HIDDEN_NEURONS_NO = 15
OUTPUT_SIZE = 4 #for lunar it should be size 4, for cartPole 2
LAYERS_NO = 3
VAR_NO = LAYERS_NO*2
LEARNING_RATE = 1e-2
TOTAL_EPISODES = 5000
PERIOD = 10
PLOT_PERIOD = 100
SAVE_PERIOD = 100
DO_NORMALIZE = False    #TODO: change the name...
WEIGHTS_FILE = './lunar_weights.pkl'
LOAD = False

ENVIRONMENT = env_d
env = gym.make(ENVIRONMENT)
env.reset()



#states - a 4XT matrix, holds all the states from up to the T step
       #notice it can be a matrix or a vector
#y = agent(observations)     #TODO: right now y = 0 becaue that's what the function returns... do i really want to do it in a function? i can tale the rest of the code after the y out.

def InitializeVarXavier(var_name,var_shape):
    return tf.get_variable(name=var_name, shape=var_shape,
                    initializer=tf.contrib.layers.xavier_initializer())

def InitializeVarRandomNormal(var_name,var_shape):
    return tf.Variable(tf.random_normal(var_shape))

#TODO: delete this copy shit later

'''
def weight_variable(name, shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.get_variable(name, initializer=initial, dtype=tf.float32)

def bias_variable(name, shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.get_variable(name, initializer=initial, dtype=tf.float32)

'''

#here we choose how to initialize
initializeW = InitializeVarXavier
initializeb = InitializeVarXavier


# Defining the agent
# placeholder for the input
observations = tf.placeholder(tf.float32, [None, INPUT_SIZE])
W1 = initializeW("W1",[INPUT_SIZE, HIDDEN_NEURONS_NO])
b1 = initializeb("b1",[HIDDEN_NEURONS_NO])
if (LAYERS_NO == 3):
    W2 = initializeW("W2",[HIDDEN_NEURONS_NO, HIDDEN_NEURONS_NO])
    b2 = initializeb("b2",[HIDDEN_NEURONS_NO])
W3 = initializeW("W3",[HIDDEN_NEURONS_NO, OUTPUT_SIZE])
b3 = initializeb("b3",[OUTPUT_SIZE])


#first,second and third layer computations
h1 = tf.nn.tanh(tf.matmul(observations, W1) + b1)
if (LAYERS_NO == 3):
    h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)
    y = tf.nn.softmax(tf.matmul(h2, W3) + b3)  # TODO: there is some problem with the softmax
else:
    y = tf.nn.softmax(tf.matmul(h1, W3) + b3)  # TODO: there is some problem with the softmax

tvars = tf.trainable_variables()

# rewards sums from k=t to T:
rewards_arr = tf.placeholder(tf.float32, [1,None])
# actions - a mask matrix which filters ys result accordint to the actions that were chosen. see boolean mask here: https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops/slicing_and_joining#boolean_mask
actions_mask = tf.placeholder(tf.bool, [None, OUTPUT_SIZE])
filtered_actions = tf.boolean_mask(y, actions_mask)  # should return a T size vector with correct (chosen) action values
pi = tf.log(filtered_actions)
#grad = tf.reduce_sum(tf.scalar_mul(1 // tf.size(pi), tf.multiply(pi,rewards_arr)))
grad_step = tf.divide(tf.reduce_sum(tf.multiply(pi,rewards_arr)),tf.to_float(tf.size(pi)))
Gradients = tf.gradients(grad_step,tvars)

Gradients_holder = [tf.placeholder(tf.float32) for i in range(VAR_NO) ]
# then train the network - for each of the parameters do the GD as described in the HW.
train_step = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(Gradients_holder,tvars))
#train_step = tf.train.AdamOptimizer(1e-2).minimize(grad_step)

#TODO: answer the following questions:
'''
new:
6) maybe i need to zero somethings like he does every once in a while?

for tommorow:
 - take care of the crush where sum(pi)>1 (which crushes the multinomial function) by normalize it
 - don't forget we got more time

things i tried:
 - change the way i initialize

things i didn't try:
 - reduce mean
 - change the way i normalize
 - change the reduce-prize function
 - change the way i process the rewards

things to check tommorow:
 - why Aran entered minus sum to the gradient function?
 - this code that solves the problem: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg
 - other solutions: https://gym.openai.com/envs/LunarLander-v2
'''


#we assume here we get the array in the right order
def decrese_rewards(rewards):
    gama = 0.99
    dec_arr = np.array([gama**t for t in range(len(rewards))])
    res = np.multiply(rewards,dec_arr);
    return res

def get_empty_grads_sums():
    grads_sums = tf.trainable_variables()
    for i, val in enumerate(grads_sums):
        grads_sums[i] = 0
    return grads_sums

init = tf.global_variables_initializer()
def main(argv):
    rewards, states, actions_booleans = [], [], []
    episode_number,total_reward,running_reward,reward_for_plot,save_reward = 0,0,0,0,None
    # TODO: for debug
    '''
    action_sum = np.array([[0.0,0.0]])
    action_probs_arr = []
    '''
    #for plotting:
    rewards_per_episode = [0 for i in range(TOTAL_EPISODES//PLOT_PERIOD)]

    with tf.Session() as sess:
        saver = tf.train.Saver()
        #check if file is not empty
        if(os.path.isfile(WEIGHTS_FILE) and LOAD):
            #f = open(WEIGHTS_FILE,'rb')
            #r1 = sess.run(tvars)
            saver.restore(sess,WEIGHTS_FILE)
            #zip(sess.run(tvars),pkl.load(f))
            r2 = sess.run(tvars)
            #f.close()
            print("loaded weights successfully!")
        else:
            sess.run(init)

        #creates file if it doesn't exisits:
        if(not os.path.isfile(WEIGHTS_FILE)):
            open(WEIGHTS_FILE,'a').close()
            print("created file sucessfully!")

        obsrv = env.reset() # Obtain an initial observation of the environment
        grads_sums = get_empty_grads_sums()


        while episode_number <= TOTAL_EPISODES:
            #append relevant observation to action to states
            states.append(obsrv)
            modified_obsrv = np.reshape(obsrv, [1, INPUT_SIZE])
            '''
            temp = sess.run(tvars)      #TODO: for debug
            th1 = sess.run(h1,feed_dict={observations: modified_obsrv})
            th2 = sess.run(h2,feed_dict={observations: modified_obsrv})
            th3 = sess.run(y,feed_dict={observations: modified_obsrv})
            action_probs_arr.append(action_probs)
            '''
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(y,feed_dict={observations: modified_obsrv})
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
            #action_sum += np.array(action_probs)
            if done:
                #create the rewards sums array and reverse again
                rewards_sums = np.cumsum(rewards[::-1]);
                #normalize prizes and reverse
                rewards_sums = decrese_rewards(np.divide(rewards_sums[::-1],np.sum(rewards_sums)))
                #rewards_sums = np.divide(rewards_sums[::-1], np.sum(rewards_sums))
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
                '''
                grads = sess.run(Gradients, feed_dict={observations: states,actions_mask:actions_booleans,rewards_arr: modified_rewards_sums })
                grads_sums += np.array(grads)
                total_reward += sum(rewards)
                reward_for_plot += sum(rewards)
                running_reward = total_reward if running_reward ==0 else running_reward * 0.99 + total_reward * 0.01

                if(episode_number%SAVE_PERIOD==0):
                    if(save_reward is None or save_reward<(reward_for_plot/ SAVE_PERIOD)):
                        save_reward = (reward_for_plot/ SAVE_PERIOD)
                        saver.save(sess,WEIGHTS_FILE)
                        #save with shmickle
                        #f = open(WEIGHTS_FILE,'wb')
                        #pkl.dump(sess.run(tvars),f)
                        #f.close()
                        print('saved file successfully!')

                if(episode_number%PERIOD==0):
                    # take the train step
                    sess.run(train_step, feed_dict={Gradients_holder[i]: grads_sums[i] for i in range(VAR_NO)})
                    print ('Episode No. %f.  Total average reward %f.' % (episode_number, total_reward / PERIOD))
                    print('Average reward for episode %f.  Total average reward %f.' % (total_reward / PERIOD, running_reward / PERIOD))
                    #print("actions_sum = ", action_sum)
                    total_reward = 0
                    grads_sums = get_empty_grads_sums()
                if(episode_number%PLOT_PERIOD == 0):
                    rewards_per_episode[episode_number // PLOT_PERIOD-1] = reward_for_plot / PLOT_PERIOD
                    reward_for_plot = 0

                episode_number += 1
                rewards, states, actions_booleans = [], [], []
                # TODO: for debug
                #action_probs_arr = []
                obsrv = env.reset()

    IS_NORMALIZED = ""
    if(DO_NORMALIZE):
        IS_NORMALIZED = "normalized"
    plt.plot(rewards_per_episode)
    plt.title(ENVIRONMENT+" Rewards Average per "+str(PLOT_PERIOD)+" Episodes for "+IS_NORMALIZED+" NN with "+str(LAYERS_NO)+" layers")
    plt.savefig("..\\..\\graphs\\rewards_for_"+ENVIRONMENT+"_period_"+str(PLOT_PERIOD)+"_layers_"+str(LAYERS_NO)+"_"+IS_NORMALIZED+"_episodes_"+str(TOTAL_EPISODES)+".png")


if __name__ == '__main__':
    tf.app.run()
    # save with pickle
    #TODO: doesn't work...
    #f = open("tvars.pkl", "wb")
    #pickle.dump(tvars, f)
    #f.close()
