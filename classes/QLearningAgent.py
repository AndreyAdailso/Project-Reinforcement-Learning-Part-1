import numpy as np
import pickle

class QLearningAgent:
    """
    Q-Learning Agent class for reinforcement learning.

    Parameters:
    - states (list): List of possible states in the environment.
    - actions (list): List of possible actions in the environment.
    - gamma (float): Discount factor for future rewards (default: 0.9).
    - alpha (float): Learning rate (default: 0.1).
    - epsilon (float): Exploration rate (default: 0.1).
    - num_episodes (int): Number of episodes to train the agent (default: 1000).

    Methods:
    - choose_action(state): Chooses an action based on the epsilon-greedy policy.
    - train(transitions, rewards, max_steps): Trains the Q-Learning agent.
    - save(filename): Saves the Q-Learning agent to a file.
    - load(filename): Loads a Q-Learning agent from a file.
    - test(env, episodes, max_steps): Tests the Q-Learning agent in an environment.
    """

    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=1000):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.Q = {state: {action: 0 for action in actions} for state in states}

    def choose_action(self, state):
        # Escolhe a ação com base na política epsilon-greedy
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(self.Q[state], key=self.Q[state].get)

    def train(self, transitions, rewards, max_steps=10):
        # Treina o agente Q-Learning
        l_rewards = []
        total_reward = 0
        for episode in range(self.num_episodes):
            total_reward_episode = 0
            state = np.random.choice(self.states)
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state = transitions[state][action]
                reward = rewards[state][action]
                total_reward_episode += reward
                self.Q[state][action] += self.alpha * (reward + self.gamma * max(self.Q[next_state].values()) - self.Q[state][action])
                state = next_state
                if next_state == 'terminal':
                    break
            total_reward += total_reward_episode
            print(f"Episódio : {episode+1} | Recompensa Total: {total_reward}")
            l_rewards.append(total_reward_episode)
        return l_rewards
    
    

    def save(self, filename):
        # Salva o agente Q-Learning em um arquivo
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def test(self, env, episodes, max_steps=10):
        # Testa o agente Q-Learning em um ambiente
        total_rewards = []
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            for i in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    break
            total_rewards.append(total_reward)
        return total_rewards
