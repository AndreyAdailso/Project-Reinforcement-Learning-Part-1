#Comparando FMDP com Q-Learning
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
        # Realiza a iteração de valor para encontrar os valores ótimos de estado e a política ótima
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
        # Calcula o valor de uma ação em um estado com base nos valores de estado
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(state, action)
        return reward + self.gamma * V[next_state]

    def extract_policy(self, V):
        # Extrai a política ótima com base nos valores ótimos de estado
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
    
    def run_optimal_policy(self, env, policy_optimal, num_episodes, max_steps=10):
        rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            total_rewards = 0
            for i in range(max_steps):
                action = policy_optimal[state]
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
            rewards.append(episode_reward)
        return rewards

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


class Environment:
    def __init__(self) -> None:
        self.states = ['s1', 's2', 's3']
        self.actions = ['a1', 'a2']
        self.transitions = {
            's1': {'a1': 's1', 'a2': 's2'},
            's2': {'a1': 's3', 'a2': 's1'},
            's3': {'a1': 's3', 'a2': 's2'}
        }
        self.rewards = {
            's1': {'a1': -1, 'a2': 5},
            's2': {'a1': 10, 'a2': 5},
            's3': {'a1': -1, 'a2': -15}
        }

    def reset(self):
        self.current_state = np.random.choice(self.states)
        return self.current_state
    
    def step(self, action):
        next_state = self.transitions[self.current_state][action]
        reward = self.rewards[self.current_state][action]
        self.current_state = next_state
        return next_state, reward, False, {}

env = Environment()

fmdp = FMDP(env.states, env.actions, env.transitions, env.rewards)
values, policy = fmdp.value_iteration()

print("Valores ótimos de estado (V*):")
for state, value in values.items():
    print(f"{state}: {value}")

print("\nPolítica ótima:")
for state, action in policy.items():
    print(f"Estado {state}: Realizar ação {action}")

rewards_optimal = fmdp.run_optimal_policy(env, policy, 1000)

# Treinamento do agente Q-Learning

q_learning_agents = []

hyperparameters = [
    {'gamma': 0.9, 'alpha': 0.1, 'epsilon': 0.1, 'num_episodes': 1000},
    {'gamma': 0.9, 'alpha': 0.5, 'epsilon': 0.1, 'num_episodes': 1000},
    {'gamma': 0.9, 'alpha': 0.1, 'epsilon': 0.5, 'num_episodes': 1000},
    {'gamma': 0.9, 'alpha': 0.5, 'epsilon': 0.5, 'num_episodes': 1000},
    {'gamma': 0.95, 'alpha': 0.1, 'epsilon': 0.1, 'num_episodes': 1000},
    {'gamma': 0.95, 'alpha': 0.5, 'epsilon': 0.1, 'num_episodes': 1000},
    {'gamma': 0.95, 'alpha': 0.1, 'epsilon': 0.5, 'num_episodes': 1000},
    {'gamma': 0.95, 'alpha': 0.5, 'epsilon': 0.5, 'num_episodes': 1000},
]

# Treinar o Q-learning com diferentes conjuntos de hiperparâmetros
results_train = []
for params in hyperparameters:
    q_learning_agent = QLearningAgent(env.states, env.actions, **params)
    result = q_learning_agent.train(env.transitions, env.rewards)
    results_train.append(result)
    q_learning_agents.append(q_learning_agent)

# Comparar política ótima com políticas aprendidas pelo Q-learning
policy_optimal = fmdp.extract_policy(values)
for i, q_learning_agent in enumerate(q_learning_agents):
    policy_q_learning = {}
    for state in env.states:
        best_action = max(q_learning_agent.Q[state], key=q_learning_agent.Q[state].get)
        policy_q_learning[state] = best_action
    print(f"Hiperparâmetros {i+1}:")
    print("Política Ótima | Política Q-learning")
    for state in env.states:
        print(f"{state}: {policy_optimal[state]} | {policy_q_learning[state]}")
    print()

print("politica ótima: ", policy_optimal)
print("policy_q_learning: ", policy_q_learning)

# Teste melhores agentes Q-learning
results_test = []
for q_learning_agent in q_learning_agents:
    result = q_learning_agent.test(env, 100)
    results_test.append(result)

# plota os resultados
#for i in range(8):
#    plt.plot(np.cumsum(results_train[i]), label=f'Hiperparâmetros {i+1}')
plt.figure()
plt.plot(np.cumsum(results_train[0]), label='G:0.9, A:0.1, E:0.1')
plt.plot(np.cumsum(results_train[1]), label='G:0.9, A:0.5, E:0.1')
plt.plot(np.cumsum(results_train[4]), label='G:0.95, A:0.1, E:0.1')
plt.plot(np.cumsum(results_train[5]), label='G:0.95, A:0.5, E:0.1')
plt.plot(np.cumsum(rewards_optimal), label='Optimal Policy')
plt.xlabel('Episódios')
plt.ylabel('Recompensa Total')
plt.legend()
plt.title('Comparação entre Agente Q-learning com diferentes hiperparâmetros')
plt.show()

plt.figure()
plt.plot(np.cumsum(results_test[0]), label='G:0.9, A:0.1, E:0.1')
plt.plot(np.cumsum(results_test[1]), label='G:0.9, A:0.5, E:0.1')
plt.plot(np.cumsum(results_test[4]), label='G:0.95, A:0.1, E:0.1')
plt.plot(np.cumsum(results_test[5]), label='G:0.95, A:0.5, E:0.1')
plt.plot(np.cumsum(rewards_optimal), label='Optimal Policy')
plt.xlabel('Episódios')
plt.ylabel('Recompensa Total')
plt.legend()
plt.title('Testes com os agentes Treinados')
plt.show()
