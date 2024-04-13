#Comparando FMDP com Q-Learning
import numpy as np

class FMDP:
    def __init__(self, states, actions, transitions, rewards, gamma=0.9, epsilon=0.0001):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        self.epsilon = epsilon

    def get_next_state(self, state, action):
        return self.transitions[state][action]

    def get_reward(self, state, action):
        return self.rewards[state][action]

    def value_iteration(self):
        V = {s: 0 for s in self.states}
        while True:
            delta = 0
            for state in self.states:
                v = V[state]
                max_q_value = float('-inf')
                for action in self.actions:
                    q_value = self.get_action_value(state, action, V)
                    if q_value > max_q_value:
                        max_q_value = q_value
                V[state] = max_q_value
                delta = max(delta, abs(v - V[state]))
            if delta < self.epsilon:
                break
        policy = self.extract_policy(V)
        return V, policy

    def get_action_value(self, state, action, V):
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(state, action)
        return reward + self.gamma * V[next_state]

    def extract_policy(self, V):
        policy = {}
        for state in self.states:
            max_q_value = float('-inf')
            best_action = None
            for action in self.actions:
                q_value = self.get_action_value(state, action, V)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            policy[state] = best_action
        return policy


class QLearningAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=1000):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.Q = {state: {action: 0 for action in actions} for state in states}

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(self.Q[state], key=self.Q[state].get)

    def train(self, transitions, rewards, max_steps=10):
        l_rewards = []
        total_reward = 0
        for episode in range(self.num_episodes):
            state = np.random.choice(self.states)
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state = transitions[state][action]
                reward = rewards[state][action]
                total_reward += reward
                self.Q[state][action] += self.alpha * (reward + self.gamma * max(self.Q[next_state].values()) - self.Q[state][action])
                state = next_state
                if next_state == 'terminal':
                    break
            print(f"Episódio : {episode+1} | Recompensa Total: {total_reward}")
            l_rewards.append(total_reward)
        return l_rewards

# Definição do ambiente
"""states = ['s1', 's2', 's3']
actions = ['a1', 'a2']
transitions = {
    's1': {'a1': 's2', 'a2': 's3'},
    's2': {'a1': 's1', 'a2': 's3'},
    's3': {'a1': 's1', 'a2': 's2'}
}
rewards = {
    's1': {'a1': 1, 'a2': -1},
    's2': {'a1': -1, 'a2': 1},
    's3': {'a1': 0, 'a2': 0}
}
"""

states = ['s1', 's2', 's3']
actions = ['a1', 'a2']
transitions = {
    's1': {'a1': 's1', 'a2': 's2'},
    's2': {'a1': 's3', 'a2': 's1'},
    's3': {'a1': 's3', 'a2': 's2'}
}
rewards = {
    's1': {'a1': -1, 'a2': 5},
    's2': {'a1': 10, 'a2': 5},
    's3': {'a1': -1, 'a2': -15}
}

fmdp = FMDP(states, actions, transitions, rewards)
values, policy = fmdp.value_iteration()

print("Valores ótimos de estado (V*):")
for state, value in values.items():
    print(f"{state}: {value}")

print("\nPolítica ótima:")
for state, action in policy.items():
    print(f"Estado {state}: Realizar ação {action}")

# Treinamento do agente Q-Learning

q_learning_agents = []

hyperparameters = [
    {'gamma': 0.9, 'alpha': 0.1, 'epsilon': 0.1, 'num_episodes': 100},
    {'gamma': 0.9, 'alpha': 0.5, 'epsilon': 0.1, 'num_episodes': 100},
]

# Treinar o Q-learning com diferentes conjuntos de hiperparâmetros
for params in hyperparameters:
    q_learning_agent = QLearningAgent(states, actions, **params)
    q_learning_agent.train(transitions, rewards)
    q_learning_agents.append(q_learning_agent)

# Comparar política ótima com políticas aprendidas pelo Q-learning
policy_optimal = fmdp.extract_policy(values)
for i, q_learning_agent in enumerate(q_learning_agents):
    policy_q_learning = {}
    for state in states:
        best_action = max(q_learning_agent.Q[state], key=q_learning_agent.Q[state].get)
        policy_q_learning[state] = best_action
    print(f"Hiperparâmetros {i+1}:")
    print("Política Ótima | Política Q-learning")
    for state in states:
        print(f"{state}: {policy_optimal[state]} | {policy_q_learning[state]}")
    print()

print("politica ótima: ", policy_optimal)
print("policy_q_learning: ", policy_q_learning)
