import numpy as np
import tensorflow as tf

# Hyperparameter define / 하이퍼 파라미터 정의
# Learning Rate = Alpha

class DQN:

    def __init__(self, session: tf.Session, state_size: int, action_size: int, name: str="main") -> None:

        """Dueling QN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters

        Args:
            session (tf.Session): Tensorflow session
            state_size (int): Input dimension
            action_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.state_size = state_size
        self.action_size = action_size
        self.net_name = name

        self.build_model()

    def build_model(self, H_SIZE_01=512, H_SIZE_15_state = 256, H_SIZE_15_action = 256, Alpha=0.001) -> None:
        """DQN Network architecture (simple MLP)
        Args:
            h_size (int, optional): Hidden layer dimension
            Alpha (float, optional): Learning rate
        """
        # Hidden Layer 01 Size  : H_SIZE_01 = 200

        with tf.variable_scope(self.net_name):            
            # 8.	Initialize variables and placeholders = Network Initializations
            # ex). x_input = tf.placeholder(tf.float32, [None, input_size])
            # ex). y_input = tf.placeholder(tf.float32, [None, num_classes])
            self._X = tf.placeholder(dtype=tf.float32, shape= [None, self.state_size], name="input_X")
            self._Y = tf.placeholder(dtype=tf.float32, shape= [None, self.action_size], name="output_Y")
            
            # 9.	Define the model structure – Main Network
            # [Dueling DQN] Separate main network as state-net and action-net
            # [Dueling DQN] Define advantage at action network and calculate the Q-Prediction for main network with Q(s’,a) = V(s) + A(s).

            # 10.	Define the model structure – Target network
            # [Dueling DQN] Separate target network as state-net and action-net
            # [Dueling DQN] Define advantage at action network and calculate the Q-Prediction for target network with Q(s’,a) = V(s) + A(s).

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

            self._LossValue = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=Alpha)
            self._train = optimizer.minimize(self._LossValue)

    def predict(self, state: np.ndarray) -> np.ndarray:

        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        state_t = np.reshape(state, [-1, self.state_size])
        action_p = self.session.run(self._Qpred, feed_dict={self._X: state_t})
        return action_p

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)

        Returns:
            list: First element is LossValue, second element is a result from train step
        """
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._LossValue, self._train], feed)