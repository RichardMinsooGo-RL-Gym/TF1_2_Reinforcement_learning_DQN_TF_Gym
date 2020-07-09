import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque

import time
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
update_cycle = 10

memory = []
size_replay_memory = 50000
batch_size = 64

class DQN:
    def __init__(self, session: tf.Session, state_size: int, action_size: int, name: str="main") -> None:
        self.session = session
        self.state_size = state_size
        self.action_size = action_size
        self.net_name = name
        
        self.build_model()

    def build_model(self, H_SIZE_01 = 256, Alpha=0.001) -> None:
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(dtype=tf.float32, shape= [None, self.state_size], name="input_X")
            self._Y = tf.placeholder(dtype=tf.float32, shape= [None, self.action_size], name="output_Y")
            net_0 = self._X

            h_fc1 = tf.layers.dense(net_0, H_SIZE_01, activation=tf.nn.relu)
            output = tf.layers.dense(h_fc1, self.action_size)
            self._Qpred = output

            self.Loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=Alpha)
            self._train = optimizer.minimize(self.Loss)

    def predict(self, state: np.ndarray) -> np.ndarray:
        x = np.reshape(state, [-1, self.state_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self.Loss, self._train], feed)

def train_model(agent: DQN, minibatch: list) -> float:
    states      = np.vstack([batch[0] for batch in minibatch])
    actions     = np.array( [batch[1] for batch in minibatch])
    rewards     = np.array( [batch[2] for batch in minibatch])
    next_states = np.vstack([batch[3] for batch in minibatch])
    dones       = np.array( [batch[4] for batch in minibatch])

    X_batch = states
    q_value = agent.predict(states)

    y_array = rewards + discount_factor * np.max(agent.predict(next_states), axis=1) * ~dones
    q_value[np.arange(len(X_batch)), actions] = y_array

    Loss, _ = agent.update(X_batch, q_value)

    return Loss

def main():

    memory = deque(maxlen=size_replay_memory)
    progress = " "

    with tf.Session() as sess:
        agent = DQN(sess, state_size, action_size, name="main")
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)

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
                
                if epsilon > np.random.rand(1):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(agent.predict(state))

                next_state, reward, done, _ = env.step(action)

                memory.append((state, action, reward, next_state, done))

                if len(memory) > size_replay_memory:
                    memory.popleft()
                
                if progress == "Training":
                    minibatch = random.sample(memory, batch_size)
                    train_model(agent, minibatch)
                    
                    if epsilon > epsilon_min:
                        epsilon -= epsilon_decay
                    else:
                        epsilon = epsilon_min

                state = next_state
                score = ep_step

                if done or ep_step == 10000:
                    if progress == "Training":
                        episode += 1
                        scores.append(score)
                        episodes.append(episode)
                        avg_score = np.mean(scores[-min(30, len(scores)):])

                    print("episode {:>5d} / score:{:>5d} / recent 30 game avg:{:>5.1f} / epsilon :{:>1.5f}"
                              .format(episode, score, avg_score, epsilon))            
                    break

        save_path = saver.save(sess, model_path + "/model.ckpt")
        print("\n Model saved in file: %s" % save_path)

        pylab.plot(episodes, scores, 'b')
        pylab.savefig(graph_path + "/mountaincar_NIPS2013.png")

        e = int(time.time() - start_time)
        print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

    # Replay the result
        episode = 0
        scores = []
        while episode < 20:
            
            state = env.reset()
            done = False
            ep_step = 0
            
            while not done and ep_step < 1000:
                env.render()
                ep_step += 1
                q_value = agent.predict(state)
                action = np.argmax(q_value)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                score = ep_step
                
                if done or ep_step == 1000:
                    episode += 1
                    scores.append(score)
                    print("episode : {:>5d} / reward : {:>5d} / avg reward : {:>5.2f}".format(episode, score, np.mean(scores)))

if __name__ == "__main__":
    main()
