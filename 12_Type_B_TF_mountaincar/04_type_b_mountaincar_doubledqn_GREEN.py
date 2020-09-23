import tensorflow as tf
import random as ran
import numpy as np
import time, datetime
from collections import deque
import gym
import pylab
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

game_name =  sys.argv[0][:-3]

model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

learning_rate = 0.001
discount_factor = 0.99
        
epsilon_max = 1.0
epsilon_min = 0.0001
epsilon_decay = 0.0001

hidden1 = 256
target_update_cycle = 200

memory = []
size_replay_memory = 5000
batch_size = 64

X = tf.placeholder(dtype=tf.float32, shape=(None, state_size), name="input_X")
Y = tf.placeholder(dtype=tf.float32, shape=(None, action_size), name="output_Y")
dropout = tf.placeholder(dtype=tf.float32)

w_fc1 = tf.get_variable('w_fc1',shape=[state_size, hidden1]
                        ,initializer=tf.contrib.layers.xavier_initializer())
w_output = tf.get_variable('w_output',shape=[hidden1, action_size]
                        ,initializer=tf.contrib.layers.xavier_initializer())

b_fc1 = tf.Variable(tf.zeros([1],dtype=tf.float32))

_h_fc1 = tf.nn.relu(tf.matmul(X,w_fc1)+b_fc1)
h_fc1 = tf.nn.dropout(_h_fc1,dropout)
output = tf.matmul(h_fc1,w_output)

# 10.	Define the model structure – Target network
w_fc1_tgt = tf.get_variable('w_fc1_tgt',shape=[state_size, hidden1])
w_output_tgt = tf.get_variable('w_output_tgt',shape=[hidden1, action_size])
b_fc1_tgt = tf.Variable(tf.zeros([1],dtype=tf.float32))

h_fc1_tgt = tf.nn.relu(tf.matmul(X ,w_fc1_tgt)+b_fc1_tgt)
output_tgt = tf.matmul(h_fc1_tgt,w_output_tgt)

Loss = tf.reduce_sum(tf.square(Y-output))
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.01)
train = optimizer.minimize(Loss)

memory = deque(maxlen=size_replay_memory)
progress = " "

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    
    sess.run(w_fc1_tgt.assign(w_fc1))
    sess.run(w_output_tgt.assign(w_output))
    sess.run(b_fc1_tgt.assign(b_fc1))

    avg_score = 10000
    episode = 0
    episodes, scores = [], []
    epsilon = epsilon_max
    start_time = time.time()
    
    while time.time() - start_time < 10*60 and avg_score > 200:

        state = env.reset()
        score = 0
        done = False
        ep_step = 0

        while not done and ep_step < 10000 :

            if len(memory) < size_replay_memory:
                progress = "Exploration"            
            else:
                progress = "Training"

            #env.render()
            ep_step += 1

            state = np.reshape(state,[1,state_size])
            q_value = sess.run(output, feed_dict={X:state, dropout: 1})

            if epsilon > np.random.rand(1):
                action = env.action_space.sample()
            else:
                action = np.argmax(q_value)

            next_state, reward, done, _ = env.step(action)

            memory.append([state, action, reward, next_state, done])

            if len(memory) > size_replay_memory:
                memory.popleft()

            if progress == "Training":
                for minibatch in ran.sample(memory, batch_size):

                    state_b, action_b, reward_b, next_state_b, done_b = minibatch
                    
                    q_value = sess.run(output, feed_dict={X: state_b, dropout: 1})

                    if done_b:
                        q_value[0, action_b] = -100
                    else:
                        next_state_b = np.reshape(next_state_b,[1,state_size])
                        tgt_q_value_next, q_value_next = sess.run([output_tgt,output], feed_dict={X: next_state_b, dropout:1})
                        q_value[0, action_b] = reward_b + discount_factor * tgt_q_value_next[0, np.argmax(q_value_next)]

                    _, loss = sess.run([train, Loss], feed_dict={X: state_b, Y: q_value, dropout:1})

                if epsilon > epsilon_min:
                    epsilon -= epsilon_decay
                else:
                    epsilon = epsilon_min

            state = next_state

            if done or ep_step % target_update_cycle == 0:
                sess.run(w_fc1_tgt.assign(w_fc1))
                sess.run(w_output_tgt.assign(w_output))
                sess.run(b_fc1_tgt.assign(b_fc1))

            if done or ep_step == 10000:
                if progress == "Training":
                    episode += 1
                    scores.append(ep_step)
                    episodes.append(episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])

                print("episode {:>5d} / score:{:>5d} / recent 30 game avg:{:>5.1f} / epsilon :{:>1.5f}"
                          .format(episode, ep_step, avg_score, epsilon))
                break

    save_path = saver.save(sess, model_path + "/model.ckpt")
    print("\n Model saved in file: %s" % save_path)

    pylab.plot(episodes, scores, 'b')
    pylab.savefig(graph_path + "/mountaincar_doubledqn.png")

    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

# Replay the result
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, model_path+ "/model.ckpt")
    print("Play Mountain Car!")
    
    episode = 0
    scores = []
    
    while episode < 20:
        
        state = env.reset()
        done = False
        ep_step = 0
        
        while not done and ep_step < 1000:
            # Plotting
            env.render()
            ep_step += 1
            state = np.reshape(state, [1, state_size])
            q_value = sess.run(output, feed_dict={X:state, dropout: 1})
            action = np.argmax(q_value)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score = ep_step
            if done or ep_step == 1000:
                episode += 1
                scores.append(score)
                print("episode : {:>5d} / reward : {:>5d} / avg reward : {:>5.2f}".format(episode, score, np.mean(scores)))