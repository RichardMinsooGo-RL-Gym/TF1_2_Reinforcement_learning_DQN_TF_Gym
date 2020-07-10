import numpy as np
import tensorflow as tf

class DQN:

    def __init__(self, session: tf.Session, state_size: int, action_size: int, name: str="main") -> None:

        self.session = session
        self.state_size = state_size
        self.action_size = action_size
        self.net_name = name

        self.build_model()

    def build_model(self, H_SIZE_01=512,Alpha=0.001) -> None:
        
        with tf.variable_scope(self.net_name):            

            self._X = tf.placeholder(dtype=tf.float32, shape= [None, self.state_size], name="input_X")
            self._Y = tf.placeholder(dtype=tf.float32, shape= [None, self.action_size], name="output_Y")
            
            net_0 = self._X

            net_1 = tf.layers.dense(net_0, H_SIZE_01, activation=tf.nn.relu)
            net_16 = tf.layers.dense(net_1, self.action_size)
            self._Qpred = net_16

            self._LossValue = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=Alpha)
            self._train = optimizer.minimize(self._LossValue)

    def predict(self, state: np.ndarray) -> np.ndarray:

        state_t = np.reshape(state, [-1, self.state_size])
        action_p = self.session.run(self._Qpred, feed_dict={self._X: state_t})
        return action_p

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._LossValue, self._train], feed)
