import unittest
import gym
import numpy as np
import tensorflow as tf
from replay import Replay
from agent import DQN_Agent
from network import Network

class TestReplay(unittest.TestCase):
    def setUp(self):
        ''' Create class instance and Gym environment instance '''
        self.memory = Replay(4)
        self.env = gym.make('CartPole-v0')

    def test_burn_memory(self):
        ''' Test to check burn_memory functionality '''
        self.memory.burn_memory(self.env, 2)
        self.assertEqual(len(self.memory.store), 2)

    def test_replace(self):
        ''' Test to check replacement of old transition tuples after crossing capacity '''
        self.memory.burn_memory(self.env, 2)
        state = self.env.reset()
        for _ in range(4):
            random_action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(random_action)
            self.memory.add_to_memory((next_state, reward, state, done))
            if done:
                state = self.env.reset()
            else:
                state = next_state
        self.assertEqual(len(self.memory.store), self.memory.capacity)

    def test_sample(self):
        ''' Test to check sampling function of replay memory '''
        self.memory.burn_memory(self.env, 3)
        batch = self.memory.sample_from_memory(2)
        self.assertEqual(len(batch),2)

class TestNetwork(unittest.TestCase):
    def setUp(self):
        ''' Create class instance and Gym environment instance '''
        self.env = gym.make('CartPole-v0')
        self.Qnet = Network(self.env.observation_space.shape[0], self.env.action_space.n, 'test')

    def test_Q_net_predict(self):
        ''' Test to check Q-network predict function '''
        state = self.env.reset()
        self.assertEqual(self.Qnet.predict(state.reshape(1,-1)).shape[1],self.env.action_space.n)

    def test_Q_net_fit(self):
        ''' Test to check Q-network fit function with dummy inputs and targets '''
        batch = 8
        states = np.random.randn(batch, self.env.observation_space.shape[0])
        targets = np.random.randn(batch)
        actions = np.random.choice(self.env.action_space.n,batch)
        actions_selected = np.vstack([np.arange(batch),actions]).T
        self.Qnet.fit(states, targets, actions_selected)


if __name__ == '__main__':
    unittest.main()