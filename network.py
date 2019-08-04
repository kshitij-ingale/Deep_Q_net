'''
Script to generate neural networks for estimating Q-value
'''
import tensorflow as tf
from config import Network_parameters

class Network:
    def __init__(self, state_dim, action_dim, name, duel=False):
        input_layer = tf.keras.Input(shape=(state_dim,))
        x = tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer="he_uniform") (input_layer)
        x = tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer="he_uniform")(x)
        if duel:
            v = tf.keras.layers.Dense(1,)(x)
            a = tf.keras.layers.Dense(action_dim)(x)
            output_layer = v + (a - tf.reduce_mean(a))

        else:
            output_layer = tf.keras.layers.Dense(action_dim)(x)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name=name)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=Network_parameters.learning_rate)

    def grad(self, inputs, targets, filters):
        def loss(input_tensor, target, filter_output):
            output_qval = self.model(input_tensor)
            q_val_action = tf.gather_nd(output_qval, filter_output)
            return tf.keras.losses.mean_squared_error(q_val_action, target)

        with tf.GradientTape() as tape:
            loss_value = loss(inputs, targets, filters)
            return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def predict(self, states, batch=Network_parameters.batch_size):
        return self.model.predict(states, batch_size=batch)

    def fit(self, states, target_Q_values, selected_actions):
        loss_value, grads = self.grad(states, target_Q_values, selected_actions)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_model(self, target_model):
        tau=Network_parameters.update_smoothing
        src_vars = self.model.trainable_variables
        dest_vars = target_model.model.trainable_variables
        for s_var, d_var in zip(src_vars, dest_vars):
            d_var.assign((tau*s_var)+((1-tau)*d_var))