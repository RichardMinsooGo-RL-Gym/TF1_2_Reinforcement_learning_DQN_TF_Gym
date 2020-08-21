import tensorflow as tf
import random
import numpy as np
import time, datetime
from collections import deque
import dqn
import gym
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
update_cycle = 10

memory = []
size_replay_memory = 50000
batch_size = 64

def train_model(agent: dqn.DQN, minibatch: list) -> float:
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
        agent = dqn.DQN(sess, state_size, action_size, name="main")
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)

        avg_score = -120
        episode = 0
        episodes, scores = [], []
        epsilon = epsilon_max
        start_time = time.time()

        while time.time() - start_time < 10*60 and avg_score < -15:
            
            state = env.reset()
            score = 0
            done = False
            ep_step = 0
            score = 0
            
            while not done and ep_step < 200:   

                if len(memory) < size_replay_memory:
                    progress = "Exploration"            
                else:
                    progress = "Training"

                #env.render()
                ep_step += 1
                
                if epsilon > np.random.rand(1):
                    # action = env.action_space.sample()
                    action = np.random.randint(0, agent.action_size)

                else:
                    actions_value = agent.predict(state)
                    action = np.argmax(actions_value)

                f_action = (action-(action_size-1)/2)/((action_size-1)/4)
                # print(f_action)
                next_state, reward, done, _ = env.step(np.array([f_action]))
                
                reward /= 10
                score += reward                
                
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
            score = 0
            
            while not done and ep_step < 200:
                env.render()
                ep_step += 1
                q_value = agent.predict(state)
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

if __name__ == "__main__":
    main()
