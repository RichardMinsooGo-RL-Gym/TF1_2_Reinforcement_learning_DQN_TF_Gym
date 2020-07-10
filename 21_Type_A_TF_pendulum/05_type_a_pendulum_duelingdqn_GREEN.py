import tensorflow as tf
import gym
import numpy as np
import random as ran
from collections import deque
import time
import pylab
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.framework import ops
ops.reset_default_graph()

env = gym.make('Pendulum-v0')
# env = env.unwrapped
env.seed(1)

state_size = env.observation_space.shape[0]
action_size = 25

file_name =  sys.argv[0][:-3]

model_path = "save_model/" + file_name
graph_path = "save_graph/" + file_name

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

H_SIZE_15_state = 256
H_SIZE_15_action = 256
H_SIZE_16_state = action_size
H_SIZE_16_action = action_size

memory = []
size_replay_memory = 5000
batch_size = 64

X = tf.placeholder(dtype=tf.float32, shape=(None, state_size), name="input_X")
Y = tf.placeholder(dtype=tf.float32, shape=(None, action_size), name="output_Y")
dropout = tf.placeholder(dtype=tf.float32)

w_fc1 = tf.get_variable('w_fc1',shape=[state_size, hidden1]
                        ,initializer=tf.contrib.layers.xavier_initializer())
W15_m_state       = tf.get_variable('W15_m_state',shape=[hidden1, H_SIZE_15_state]
                              ,initializer=tf.contrib.layers.xavier_initializer())
W15_m_action      = tf.get_variable('W15_m_action',shape=[hidden1, H_SIZE_15_action]
                               ,initializer=tf.contrib.layers.xavier_initializer())
W16_m_state       = tf.get_variable('W16_m_state',shape=[H_SIZE_15_state, H_SIZE_16_state]
                              ,initializer=tf.contrib.layers.xavier_initializer())
W16_m_action      = tf.get_variable('W16_m_action',shape=[H_SIZE_15_action, H_SIZE_16_action]
                               ,initializer=tf.contrib.layers.xavier_initializer())

b_fc1 = tf.Variable(tf.zeros([1],dtype=tf.float32))
B15_m_state       = tf.Variable(tf.zeros([1],dtype=tf.float32))
B15_m_action      = tf.Variable(tf.zeros([1],dtype=tf.float32))
B16_m_state       = tf.Variable(tf.zeros([1],dtype=tf.float32))
B16_m_action      = tf.Variable(tf.zeros([1],dtype=tf.float32))

_h_fc1 = tf.nn.relu(tf.matmul(X,w_fc1)+b_fc1)
h_fc1 = tf.nn.dropout(_h_fc1,dropout)

_LAY15_m_state    = tf.nn.relu(tf.matmul(h_fc1,W15_m_state)+B15_m_state)
LAY15_m_state     = tf.nn.dropout(_LAY15_m_state,dropout)
_LAY15_m_action   = tf.nn.relu(tf.matmul(h_fc1,W15_m_action)+B15_m_action)
LAY15_m_action    = tf.nn.dropout(_LAY15_m_action,dropout)

_output_state    = tf.matmul(LAY15_m_state,W16_m_state) + B16_m_state
output_state     = tf.nn.dropout(_output_state,dropout)
_output_action   = tf.matmul(LAY15_m_action,W16_m_action) + B16_m_action
output_action    = tf.nn.dropout(_output_action,dropout)

output_advantage = tf.subtract(output_action, tf.reduce_mean(output_action))

output = tf.add(output_state, output_advantage)

w_fc1_tgt = tf.get_variable('w_fc1_tgt',shape=[state_size, hidden1])
W15_t_state       = tf.get_variable('W15_t_state',shape=[hidden1, H_SIZE_15_state])
W15_t_action      = tf.get_variable('W15_t_action',shape=[hidden1, H_SIZE_15_action])
W16_t_state       = tf.get_variable('W16_t_state',shape=[H_SIZE_15_state, H_SIZE_16_state])
W16_t_action      = tf.get_variable('W16_t_action',shape=[H_SIZE_15_action, H_SIZE_16_action])

b_fc1_tgt = tf.Variable(tf.zeros([1],dtype=tf.float32))
B15_t_state       = tf.Variable(tf.zeros([1],dtype=tf.float32))
B15_t_action      = tf.Variable(tf.zeros([1],dtype=tf.float32))
B16_t_state       = tf.Variable(tf.zeros([1],dtype=tf.float32))
B16_t_action      = tf.Variable(tf.zeros([1],dtype=tf.float32))

h_fc1_tgt = tf.nn.relu(tf.matmul(X ,w_fc1_tgt)+b_fc1_tgt)

LAY15_t_state     = tf.nn.relu(tf.matmul(h_fc1_tgt,W15_t_state)+B15_t_state)
LAY15_t_action    = tf.nn.relu(tf.matmul(h_fc1_tgt,W15_t_action)+B15_t_action)

output_tgt_state     = tf.matmul(LAY15_t_state,W16_t_state) + B16_t_state
output_tgt_action    = tf.matmul(LAY15_t_action,W16_t_action) + B16_t_action

output_tgt_advantage = tf.subtract(output_tgt_action, tf.reduce_mean(output_tgt_action))

output_tgt = tf.add(output_tgt_state, output_tgt_advantage)

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
    sess.run(W15_t_state.assign(W15_m_state))
    sess.run(W15_t_action.assign(W15_m_action))
    sess.run(W16_t_state.assign(W16_m_state))
    sess.run(W16_t_action.assign(W16_m_action))
    
    sess.run(b_fc1_tgt.assign(b_fc1))
    sess.run(B15_t_state.assign(B15_m_state))
    sess.run(B15_t_action.assign(B15_m_action))
    sess.run(B16_t_state.assign(B16_m_state))
    sess.run(B16_t_action.assign(B16_m_action))
    avg_score = -120
    episode = 0
    episodes, scores = [], []
    epsilon = epsilon_max
    start_time = time.time()
    
    while time.time() - start_time < 5*60 and avg_score < -15: 

        state = env.reset()
        score = 0
        done = False
        ep_step = 0
        score = 0

        while not done and ep_step < 200 :

            if len(memory) < size_replay_memory:
                progress = "Exploration"            
            else:
                progress = "Training"

            #env.render()
            ep_step += 1

            state = np.reshape(state,[1,state_size])
            q_value = sess.run(output, feed_dict={X:state, dropout: 1})

            if epsilon > np.random.rand(1):
                # action = env.action_space.sample()
                action = np.random.randint(0, action_size)
            else:
                action = np.argmax(q_value)

            f_action = (action-(action_size-1)/2)/((action_size-1)/4)
            # print(f_action)
            next_state, reward, done, _ = env.step(np.array([f_action]))

            reward /= 10
            score += reward

            memory.append([state, action, reward, next_state, done, ep_step])

            if len(memory) > size_replay_memory:
                memory.popleft()

            if progress == "Training":
                minibatch = ran.sample(memory, batch_size)
                for states, actions, rewards, next_states, dones ,ep_steps in minibatch:
                    
                    q_value = sess.run(output, feed_dict={X: states, dropout: 1})

                    if dones:
                        if ep_steps < env.spec.timestep_limit :
                            q_value[0, actions] = -100
                    else:
                        next_states = np.reshape(next_states,[1,state_size])
                        tgt_q_value_next = sess.run(output_tgt, feed_dict={X: next_states, dropout:1})
                        q_value[0, actions] = rewards + discount_factor * np.max(tgt_q_value_next)

                    _, loss = sess.run([train, Loss], feed_dict={X: states, Y: q_value, dropout:1})

                if epsilon > epsilon_min:
                    epsilon -= epsilon_decay
                else:
                    epsilon = epsilon_min

            state = next_state

            if done or ep_step % target_update_cycle == 0:
                sess.run(w_fc1_tgt.assign(w_fc1))
                sess.run(W15_t_state.assign(W15_m_state))
                sess.run(W15_t_action.assign(W15_m_action))
                sess.run(W16_t_state.assign(W16_m_state))
                sess.run(W16_t_action.assign(W16_m_action))
                sess.run(b_fc1_tgt.assign(b_fc1))
                sess.run(B15_t_state.assign(B15_m_state))
                sess.run(B15_t_action.assign(B15_m_action))
                sess.run(B16_t_state.assign(B16_m_state))
                sess.run(B16_t_action.assign(B16_m_action))            

            if done or ep_step == 200:
                if progress == "Training":
                    episode += 1
                    scores.append(score)
                    episodes.append(episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])

                print("episode {:>5d} / score:{:>5.1f} / recent 30 game avg:{:>5.1f} / epsilon :{:>1.5f}"
                          .format(episode, score, avg_score, epsilon))            
                break

    save_path = saver.save(sess, model_path + "/model.ckpt")
    print("\n Model saved in file: %s" % save_path)

    pylab.plot(episodes, scores, 'b')
    pylab.savefig(graph_path + "/pendulum_Nature2015.png")

    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

# Replay the result
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, model_path+ "/model.ckpt")
    print("Play Cartpole!")
    
    episode = 0
    scores = []
    
    while episode < 20:
        
        state = env.reset()
        done = False
        ep_step = 0
        score = 0
        
        while not done and ep_step < 200:
            # Plotting
            env.render()
            ep_step += 1
            state = np.reshape(state, [1, state_size])
            q_value = sess.run(output, feed_dict={X:state, dropout: 1})
            action = np.argmax(q_value)
            f_action = (action-(action_size-1)/2)/((action_size-1)/4)
            # print(f_action)
            next_state, reward, done, _ = env.step(np.array([f_action]))
            reward /= 10
            score += reward
            state = next_state
            
            if done or ep_step == 200:
                episode += 1
                scores.append(score)
                print("episode : {:>5d} / score : {:>5.1f} / avg reward : {:>5.1f}".format(episode, score, np.mean(scores)))