#Comparando FMDP com Q-Learning
import numpy as np
import pickle
import matplotlib.pyplot as plt

from classes.Environment import Environment
from classes.FMDP import FMDP
from classes.QLearningAgent import QLearningAgent

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

# plota os resultados dos treinamentos
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

# plota os resultados dos testes
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

# plota a recompensa média acumulada para cada agente
plt.figure()
for i in range(8):
    plt.plot(np.cumsum(results_train[i])/np.arange(1, len(results_train[i])+1), label=f'Hiperparâmetros {i+1}')
plt.xlabel('Iterações')
plt.ylabel('Recompensa Média Acumulada')
plt.legend()
plt.title('Convergência dos Agentes Q-learning com diferentes hiperparâmetros')
plt.show()
