'''
Script for functions related to replay memory
'''
import random
from config import Network_parameters, Replay_parameters

class Replay:
    def __init__(self, capacity=Replay_parameters.capacity):
        self.capacity = capacity
        self.current_count = 0
        self.store = []

    def add_to_memory(self, transition_tuple):
        """
        Adds transition tuple to replay memory
        Args:
            transition_tuple: State transition tuple to be added

        Returns:
            None
        """
        if self.current_count<self.capacity:
            self.store.append(transition_tuple)
        else:
            self.store[self.current_count%self.capacity] = transition_tuple
        self.current_count += 1

    def sample_from_memory(self, batch_size=Network_parameters.batch_size):
        """
        Sample batch from replay memory
        Args:
            batch_size: Size of batch to be sampled

        Returns:
            Batch of state transition tuples
        """
        return random.sample(self.store, batch_size)


    def burn_memory(self, env, burn_in=Replay_parameters.burn_in_episodes):
        """
        Burn transition tuples to memory to initialize memory
        Args:
            env: Gym environment instance
            burn_in: Number of transition tuples to be initialized

        Returns:
            None
        """
        state = env.reset()
        while self.current_count < burn_in:
            random_action = env.action_space.sample()
            next_state, reward, done, _ = env.step(random_action)
            self.add_to_memory((next_state, reward, state, random_action, done))
            if done:
                state = env.reset()
            else:
                state = next_state
