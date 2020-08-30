import random
import numpy as np
import time, datetime
from collections import deque
import gym
import pylab
import sys
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()
import tensorflow as tf

env = gym.make('CartPole-v1')

# get size of state and action from environment
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

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQN:
    """ Implementation of deep q learning algorithm """   
    def __init__(self, session: tf.Session, state_size: int, action_size: int, name: str="main") -> None:
        
        self.render = False
        # get size of state and action
        self.session = session
        self.progress = " "
        self.state_size = state_size
        self.action_size = action_size
        
        # train time define
        self.training_time = 5*60
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        # final value of epsilon
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.0005
        self.epsilon = self.epsilon_max
        
        self.step = 0
        self.score = 0
        self.episode = 0
        
        self.hidden1, self.hidden2 = 64, 64
        
        self.ep_trial_step = 500
        
        # Parameter for Experience Replay
        self.size_replay_memory = 5000
        self.batch_size = 64
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        self.net_name = name
        self.model = self.build_model()
        
    # def build_model(self, H_SIZE_01 = 256, H_SIZE_02 = 256, H_SIZE_03 = 256, self.learning_rate=0.001) -> None:
    def build_model(self):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(dtype=tf.float32, shape= [None, self.state_size], name="input_X")
            self._Y = tf.placeholder(dtype=tf.float32, shape= [None, self.action_size], name="output_Y")
            net_0 = self._X

            net = tf.layers.dense(net_0, self.hidden1, activation=tf.nn.relu)
            net_16 = tf.layers.dense(net, self.action_size)
            self._Qpred = net_16

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

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        # choose an action_arr epsilon greedily
        action_arr = np.zeros(self.action_size)
        action = 0
        
        if random.random() < self.epsilon:
            # print("----------Random action_arr----------")
            action = random.randrange(self.action_size)
            action_arr[action] = 1
        else:
            # Predict the reward value based on the given state
            Q_value = self.predict(state)
            action = np.argmax(Q_value[0])
            action_arr[action] = 1
            
        return action_arr, action

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self,state,action,reward,next_state,done):
        #in every action put in the memory
        self.memory.append((state,action,reward,next_state,done))
        
        while len(self.memory) > self.size_replay_memory:
            self.memory.popleft()
    
def train_model(agent):
    minibatch = random.sample(agent.memory, agent.batch_size)
    for state,action,reward,next_state, done in minibatch:
        q_update = reward
        if not done:
            q_update = (reward + agent.discount_factor*np.amax(agent.predict(next_state)[0]))
        q_values = agent.predict(state)
        q_values[0][action] = q_update
        agent.update(state,q_values)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay
    else :
        agent.epsilon = agent.epsilon_min
        
def main():

    # with tf.Session() as sess:
    sess = tf.Session()
    agent = DQN(sess, state_size, action_size, name="main")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)

    avg_score = 0
    episodes, scores = [], []
    
    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    print("\n\n",game_name, "-game start at :",display_time,"\n")
    start_time = time.time()

    while time.time() - start_time < agent.training_time and avg_score < 490:

        state = env.reset()
        done = False
        agent.score = 0
        ep_step = 0
        state = np.reshape(state, [1, state_size])
        while not done and ep_step < agent.ep_trial_step:
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"            
            else:
                agent.progress = "Training"

            ep_step += 1
            agent.step += 1
            
            if agent.render:
                env.render()
                
            action_arr, action = agent.get_action(state)
            
            # run the selected action and observe next state and reward
            next_state, reward, done, _ = env.step(action)
            
            next_state = np.reshape(next_state, [1, state_size])
            
            if done:
                reward = -100
            
            # store the transition in memory
            agent.append_sample(state, action, reward, next_state, done)

            # update the old values
            state = next_state
            # only train if done observing
            if agent.progress == "Training":
                # Training!
                train_model(agent)

            agent.score = ep_step
            
            if done or ep_step == agent.ep_trial_step:
                if agent.progress == "Training":
                    agent.episode += 1
                    scores.append(agent.score)
                    episodes.append(agent.episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])
                print('episode :{:>6,d}'.format(agent.episode),'/ ep step :{:>5,d}'.format(ep_step), \
                      '/ time step :{:>7,d}'.format(agent.step),'/ status :', agent.progress, \
                      '/ epsilon :{:>1.4f}'.format(agent.epsilon),'/ last 30 avg :{:> 4.1f}'.format(avg_score) )
                break

    save_path = saver.save(sess, model_path + "/model.ckpt")
    print("\n Model saved in file: %s" % save_path)

    pylab.plot(episodes, scores, 'b')
    pylab.savefig(graph_path + "/cartpole_NIPS2013.png")

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

        while not done and ep_step < 500:
            env.render()
            ep_step += 1
            q_value = agent.predict(state)
            action = np.argmax(q_value)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score = ep_step

            if done or ep_step == 500:
                episode += 1
                scores.append(score)
                print("episode : {:>5d} / reward : {:>5d} / avg reward : {:>5.2f}".format(episode, score, np.mean(scores)))

if __name__ == "__main__":
    main()
