'''
Script to train DQN agent for Gym environment
'''
import argparse, gym, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from network import Network
from replay import Replay
from config import Network_parameters, Training_parameters, Directories

class DQN_Agent:

    def __init__(self, parameters):
        # Gym environment parameters
        self.env_name = parameters.environment_name
        self.env = gym.make(self.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        # Training parameters
        self.discount = Training_parameters.discount
        self.train_episodes = parameters.train_episodes
        self.test_episodes = Training_parameters.test_episodes
        self.test_frequency = Training_parameters.test_frequency
        self.render_decision = parameters.render_decision
        self.render_frequency = Training_parameters.render_frequency
        # Replay memory parameters
        self.memory = Replay()
        self.memory.burn_memory(self.env)
        # Q-networks parameters
        self.Q_net = Network(self.state_dim, self.action_dim, Network_parameters.Q_net_var_scope, parameters.duel)
        self.target_Q_net = Network(self.state_dim, self.action_dim, Network_parameters.target_Q_net_var_scope, parameters.duel)
        self.update_target_frequency = Training_parameters.update_target_frequency
        self.double = parameters.double

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        """
        Returns action as per epsilon-greedy policy
        :param q_values: Q-values for the possible actions
        :param epsilon: Parameter to define exploratory action probability
        :return: action: Action selected by agent as per epsilon-greedy policy
        """
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(q_values)

    def greedy_policy(self, q_values):
        '''
        Returns action as per greedy policy

        Parameters:
        q_values: Q-values for the possible actions

        Output:
        Action selected by agent as per greedy policy corresponding to maximum Q-value
        '''
        return np.argmax(q_values)

    def train(self):
        performance = []
        # Setup video rendering for Gym environment
        if self.render_decision:
            f = lambda X: X % self.render_frequency == 0
            self.env.render()
            video_save_path = f'{Directories.output}Video_DQN_{self.env_name}/'
            self.env = gym.wrappers.Monitor(self.env, video_save_path, video_callable=f, force=True)
            self.env.reset()

        for episode in range(self.train_episodes):
            state = self.env.reset()
            done = False
            while not done:
                # Perform an action in environment and add to replay memory
                Q_values = self.Q_net.predict(state.reshape(-1, self.state_dim))
                epsilon = 0.05 - episode * ((0.05 - 0.005) / self.train_episodes) # Anneal epsilon to 0.005
                action = self.epsilon_greedy_policy(Q_values, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.add_to_memory((next_state, reward, state, action, done))

                # Sample batch from memory and train model
                batch = self.memory.sample_from_memory()
                batch_next_state, batch_reward, batch_state, batch_action, check_if_terminal = map(np.array,
                                                                                                   zip(*batch))
                check_if_not_terminal = np.invert(check_if_terminal)
                if self.double:
                    Q_next = self.Q_net.predict(batch_next_state.reshape(-1, self.state_dim))
                    next_actions = np.argmax(Q_next, axis=1)
                    next_actions_indices = np.vstack([np.arange(Network_parameters.batch_size), next_actions]).T
                    target_Q_next_all_actions = self.target_Q_net.predict(batch_next_state.reshape(-1, self.state_dim))
                    targets = batch_reward + check_if_not_terminal * self.discount *tf.gather_nd(target_Q_next_all_actions, next_actions_indices)
                else:
                    target_Q_next = self.target_Q_net.predict(batch_next_state.reshape(-1, self.state_dim))
                    targets = batch_reward + check_if_not_terminal*self.discount * np.max(target_Q_next, axis=1)
                actions_selected = np.vstack([np.arange(Network_parameters.batch_size), batch_action]).T
                self.Q_net.fit(batch_state, targets, actions_selected)

            # Update target model as per update frequency
            if episode % self.update_target_frequency == 0:
                self.Q_net.update_target_model(self.target_Q_net)

            # Test policy as per test frequency
            if episode % self.test_frequency == 0:
                test_rewards, test_std = self.test()
                print(f'After {episode} episodes, mean test reward is {test_rewards} with std of {test_std}')
                performance.append((test_rewards, test_std))
        return performance

    def test(self):
        rewards = []
        for test_episode in range(self.test_episodes):
            curr_episode_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                action = self.greedy_policy(self.Q_net.predict(state.reshape(1, -1)))
                next_state, reward, done, _ = self.env.step(action)
                curr_episode_reward += reward
                if done:
                    state = self.env.reset()
                else:
                    state = next_state
            rewards.append(curr_episode_reward)
        rewards = np.array(rewards)
        return np.mean(rewards), np.std(rewards)


def parse_arguments():
    ''' Parse command line arguments using argparse'''
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--t', dest='train_decision', default=True, type=bool,
                        help='Train the agent')
    parser.add_argument('--e', dest='environment_name', default='CartPole-v0', type=str,
                        help='Gym Environment')
    parser.add_argument('--r', dest='render_decision', default=False, type=bool,
                        help='Render the environment')
    parser.add_argument('--db', dest='double', default=False, type=bool,
                        help='Use double DQN')
    parser.add_argument('--du', dest='duel', default=False, type=bool,
                        help='Use dueling networks')
    parser.add_argument('--ep', dest='train_episodes', default=10000, type=int,
                        help='Number of Training episodes')
    args = parser.parse_args()
    return args


def main():
    ''' Trains the DQN agent and evaluates performance'''
    args = parse_arguments()
    if args.train_decision:
        dqn = DQN_Agent(args)
        training_performance = dqn.train()

    r, s = dqn.test()
    print(f'The reward is {r} with a deviation of {s}')

    dqn.env.close()

    rewards, std = zip(*training_performance)
    episodes = list(range(0,args.train_episodes,Training_parameters.test_frequency))
    plt.errorbar(x=episodes, y=rewards, yerr=std)
    plt.savefig(f'{Directories.output}performance.png')
    plt.show()

if __name__ == '__main__':
    main()
