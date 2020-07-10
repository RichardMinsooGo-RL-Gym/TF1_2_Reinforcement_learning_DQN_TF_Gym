import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
from typing import List
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

target_update_cycle = 200

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQN:
    """ Implementation of deep q learning algorithm """   
    def __init__(self, session: tf.Session, state_size: int, action_size: int, name: str="main") -> None:
        
        #HyperParameters
        self.session = session
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.95
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 30,30
        self.size_replay_memory = 50000
        self.batch_size = 32
        self.epsilon_max = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.997
        self.epsilon = self.epsilon_max
        
        #Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        self.net_name = name
        self.model = self.build_model()
        
    def build_model(self, H_SIZE_01=200, H_SIZE_15_state = 200, H_SIZE_15_action = 200, Alpha=0.001) -> None:
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(dtype=tf.float32, shape= [None, self.state_size], name="input_X")
            self._Y = tf.placeholder(dtype=tf.float32, shape= [None, self.action_size], name="output_Y")
            self.dropout = tf.placeholder(dtype=tf.float32)
            H_SIZE_16_state = self.action_size
            H_SIZE_16_action = self.action_size
            
            net_0 = self._X

            net_1 = tf.layers.dense(net_0, H_SIZE_01, activation=tf.nn.relu)
            net_15_state = tf.layers.dense(net_1, H_SIZE_15_state, activation=tf.nn.relu)
            net_15_action = tf.layers.dense(net_1, H_SIZE_15_action, activation=tf.nn.relu)
            
            net_16_state = tf.layers.dense(net_15_state, H_SIZE_16_state)
            net_16_action = tf.layers.dense(net_15_action, H_SIZE_16_action)
            
            net16_advantage = tf.subtract(net_16_action, tf.reduce_mean(net_16_action))
            
            Q_prediction = tf.add(net_16_state, net16_advantage)
            
            self._Qpred = Q_prediction

            self.Loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
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
    
    def get_action(self,state):
        #Exploration vs Exploitation
        if np.random.rand() <= self.epsilon:
            # action = random.randrange(self.action_size)
            action = np.random.randint(0, self.action_size)
            return action
        
        q_values  = self.predict(state)
        
        return np.argmax(q_values[0])
    
    def append_sample(self,state,action,reward,next_state,done):
        #in every action put in the memory
        self.memory.append((state,action,reward,next_state,done))

def Copy_Weights(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder    
    
def train_model(agent, target_agent):
    #When the memory is filled up take a batch and train the network
    if len(agent.memory) < agent.size_replay_memory:
        return

    minibatch = random.sample(agent.memory, agent.batch_size)
    for state,action,reward,next_state, done in minibatch:
        q_update = reward
        if not done:
            #Obtain the Q' values by feeding the new state through our network
            q_update = (reward + agent.discount_factor*np.amax(target_agent.predict(next_state)[0]))
        q_values = agent.predict(state)
        q_values[0][action] = q_update
        agent.update(state,q_values)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

def main():

    progress = " "

    with tf.Session() as sess:
        agent = DQN(sess, state_size, action_size, name="main")
        target_agent = DQN(sess, state_size, action_size, name="target")
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)

        # initial copy q_net -> target_net
        copy_ops = Copy_Weights(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        avg_score = -120
        episode = 0
        episodes, scores = [], []
        start_time = time.time()

        while time.time() - start_time < 5*60 and avg_score < -15:
            
            state = env.reset()
            score = 0
            done = False
            ep_step = 0
            rewards = 0
            state = np.reshape(state, [1, state_size])
            while not done and ep_step < 200 :

                if len(agent.memory) < agent.size_replay_memory:
                    progress = "Exploration"            
                else:
                    progress = "Training"

                #env.render()
                ep_step += 1
                
                action = agent.get_action(state)

                f_action = (action-(action_size-1)/2)/((action_size-1)/4)
                # print(f_action)
                next_state, reward, done, _ = env.step(np.array([f_action]))
                
                reward /= 10
                rewards += reward 
                
                next_state = np.reshape(next_state, [1, state_size])
                agent.append_sample(state, action, reward, next_state, done)

                if len(agent.memory) > agent.size_replay_memory:
                    agent.memory.popleft()
                
                if progress == "Training":
                    train_model(agent, target_agent)
                    
                    if done or ep_step % target_update_cycle == 0:
                        sess.run(copy_ops)
                        
                state = next_state
                score = rewards

                if done or ep_step == 200:
                    if progress == "Training":
                        episode += 1
                        scores.append(score)
                        episodes.append(episode)
                        avg_score = np.mean(scores[-min(30, len(scores)):])

                    print("episode {:>5d} / score:{:>5.1f} / recent 30 game avg:{:>5.1f} / epsilon :{:>1.5f}"
                              .format(episode, score, avg_score, agent.epsilon))            
                    break

        save_path = saver.save(sess, model_path + "/model.ckpt")
        print("\n Model saved in file: %s" % save_path)

        pylab.plot(episodes, scores, 'b')
        pylab.savefig(graph_path + "/pendulum_NIPS2013.png")

        e = int(time.time() - start_time)
        print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

    # Replay the result
        episode = 0
        scores = []
        while episode < 20:
            
            state = env.reset()
            done = False
            ep_step = 0
            state = np.reshape(state, [1, state_size])
            rewards = 0
            while not done and ep_step < 200:
                env.render()
                ep_step += 1
                q_value = agent.predict(state)
                action = np.argmax(q_value)
                f_action = (action-(action_size-1)/2)/((action_size-1)/4)
                # print(f_action)
                next_state, reward, done, _ = env.step(np.array([f_action]))
                
                reward /= 10
                rewards += reward
                state = next_state
                score = rewards
                
                if done or ep_step == 1000:
                    episode += 1
                    scores.append(score)
                    print("episode : {:>5d} / reward : {:>5.1f} / avg reward : {:>5.1f}".format(episode, score, np.mean(scores)))

if __name__ == "__main__":
    main()
