'''
Script to generate neural networks for estimating Q-value
'''
import time, os
import tensorflow as tf
from tensorflow.keras.layers import Dense

from config import Network_parameters, Directories

class Network:
    def __init__(self, state_dim, action_dim, name):
        self.model = tf.keras.Sequential(
            [Dense(16, activation=tf.nn.relu, kernel_initializer="he_uniform",input_shape=(state_dim,)),
             Dense(32, activation=tf.nn.relu,kernel_initializer="he_uniform"),
             Dense(16, activation=tf.nn.relu, kernel_initializer="he_uniform"),
             Dense(action_dim)], name=name)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        train_log_dir = Directories.tensorboard
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def grad(self, inputs, targets, filters):
        def loss(input_tensor, target, filter_output):
            output_qval = self.model(input_tensor)
            q_val_action = tf.gather_nd(output_qval, filter_output)
            return tf.keras.losses.mean_squared_error(q_val_action, target)

        with tf.GradientTape() as tape:
            loss_value = loss(inputs, targets, filters)
            return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def predict(self, states):
        return self.model(states)

    def fit(self, states, target_Q_values, selected_actions, episode):
        loss_value, grads = self.grad(states, target_Q_values, selected_actions)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value,step=episode)

    def update_target_model(self, target_model):
        tau=0.9
        src_vars = self.model.trainable_variables
        dest_vars = target_model.model.trainable_variables
        for s_var, d_var in zip(src_vars, dest_vars):
            d_var.assign((tau*s_var)+((1-tau)*d_var))
