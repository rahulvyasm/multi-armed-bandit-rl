# Multi-Armed Bandit Algorithms

## Description
This repository contains Python code implementing various Multi-Armed Bandit (MAB) algorithms and their performance evaluation using different strategies. The algorithms are compared based on their action selection and reward distribution.

## Algorithms
1. **Epsilon-Greedy Algorithm**
2. **Optimistic Initialization**
3. **Upper Confidence Bound (UCB)**
4. **Gradient Bandit with Baseline**

## Code Explanation

### Initialization
We start by initializing the true action values (`q_star`) and simulating the reward distributions (`arms`) for the first problem.

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

k = 10
num_problems = 2000

q_star = np.random.normal(0, 1, (num_problems, k))
arms = [0] * k

for i in range(10):
    arms[i] = np.random.normal(q_star[0, i], 1, 2000)  # First problem as a sample

plt.figure(figsize=(12, 8))
plt.ylabel('Rewards distribution')
plt.xlabel('Actions')
plt.xticks(range(1, 11))
plt.yticks(np.arange(-5, 5, 0.5))

plt.violinplot(arms, positions=range(1, 11), showmedians=True)
plt.show()
```

### Bandit Function
The `bandit` function simulates the reward for a given action and problem instance.

```python
def bandit(action, problem):
    return np.random.normal(q_star[problem, action], 1)
```

### Simple Epsilon-Greedy Algorithm
The `simple_bandit` function implements the epsilon-greedy algorithm with action-value estimates (`Q`) and action counts (`N`).

```python
def simple_max(Q, N, t):
    return np.random.choice(np.flatnonzero(Q == Q.max()))  # Breaking ties randomly

def simple_bandit(k, epsilon, steps, initial_Q, alpha=0, argmax_func=simple_max):
    rewards = np.zeros(steps)
    actions = np.zeros(steps)

    for problem in tqdm(range(num_problems)):
        Q = np.full(k, initial_Q, dtype=float)  # Initialize action-value estimates
        N = np.zeros(k, dtype=int)  # Action counts
        best_action = np.argmax(q_star[problem])  # Optimal action for the current problem

        for t in range(steps):
            if np.random.rand() < epsilon:
                action = np.random.choice(k)  # Explore
            else:
                action = argmax_func(Q, N, t)  # Exploit

            reward = bandit(action, problem)
            rewards[t] += reward
            N[action] += 1

            if alpha > 0:  # Constant step-size
                Q[action] += alpha * (reward - Q[action])
            else:  # Sample-average method
                Q[action] += (reward - Q[action]) / N[action]

            if action == best_action:
                actions[t] += 1

    return rewards / num_problems, actions / num_problems
```

### Upper Confidence Bound (UCB) Algorithm
The `ucb` function selects actions based on the UCB formula.

```python
def ucb(Q, N, t):
    c = 2  # Exploration constant
    ucb_values = np.zeros_like(Q)

    for a in range(len(Q)):
        if N[a] == 0:
            return a  # Explore unvisited action
        ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])

    return np.random.choice(np.flatnonzero(ucb_values == ucb_values.max()))  # Break ties randomly
```

### Gradient Bandit Algorithm
The `gradient_bandit` function implements the gradient bandit algorithm with and without a baseline.

```python
def gradient_bandit(k, steps, alpha, initial_Q, is_baseline=True):
    rewards = np.zeros(steps)
    actions = np.zeros(steps)

    for problem in tqdm(range(num_problems)):
        H = np.zeros(k)  # Action preferences
        best_action = np.argmax(q_star[problem])  # Optimal action for the current problem
        baseline = 0 if is_baseline else None

        for t in range(steps):
            probabilities = np.exp(H) / np.sum(np.exp(H))
            action = np.random.choice(k, p=probabilities)

            reward = bandit(action, problem)
            rewards[t] += reward

            if is_baseline:
                baseline += (reward - baseline) / (t + 1)

            for a in range(k):
                if a == action:
                    H[a] += alpha * (reward - (baseline if is_baseline else 0)) * (1 - probabilities[a])
                else:
                    H[a] -= alpha * (reward - (baseline if is_baseline else 0)) * probabilities[a]

            if action == best_action:
                actions[t] += 1

    return rewards / num_problems, actions / num_problems
```

## Performance Evaluation
The following code evaluates and compares the performance of different strategies and visualizes the results.

```python
ep_0, ac_0 = simple_bandit(k=10, epsilon=0, steps=1000, initial_Q=0)
ep_01, ac_01 = simple_bandit(k=10, epsilon=0.01, steps=1000, initial_Q=0)
ep_1, ac_1 = simple_bandit(k=10, epsilon=0.1, steps=1000, initial_Q=0)

plt.figure(figsize=(12, 6))
plt.plot(ep_0, 'g', label='epsilon = 0')
plt.plot(ep_01, 'r', label='epsilon = 0.01')
plt.plot(ep_1, 'b', label='epsilon = 0.1')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.yticks(np.arange(0, 1, 0.1))
plt.plot(ac_0, 'g', label='epsilon = 0')
plt.plot(ac_01, 'r', label='epsilon = 0.01')
plt.plot(ac_1, 'b', label='epsilon = 0.1')
plt.legend()
plt.show()

opt_0, ac_opt_0 = simple_bandit(k=10, epsilon=0, steps=1000, initial_Q=5, alpha=0.2)

plt.figure(figsize=(12, 6))
plt.yticks(np.arange(0, 3, 0.2))
plt.plot(ac_1, 'r', label='Realistic')
plt.plot(ac_opt_0, 'b', label='Optimistic')
plt.legend()
plt.show()

ucb_2, ac_ucb_2 = simple_bandit(k=10, epsilon=0, steps=1000, initial_Q=0, argmax_func=ucb)

plt.figure(figsize=(12, 6))
plt.plot(ep_1, 'g', label='e-greedy e=0.1')
plt.plot(ucb_2, 'b', label='ucb c=2')
plt.legend()
plt.show()

sft_4, ac_sft_4 = gradient_bandit(k=10, steps=1000, alpha=0.4, initial_Q=0, is_baseline=False)
sft_4_baseline, ac_sft_4_baseline = gradient_bandit(k=10, steps=1000, alpha=0.4, initial_Q=0, is_baseline=True)

sft_1, ac_sft_1 = gradient_bandit(k=10, steps=1000, alpha=0.1, initial_Q=0, is_baseline=False)
sft_1_baseline, ac_sft_1_baseline = gradient_bandit(k=10, steps=1000, alpha=0.1, initial_Q=0, is_baseline=True)

plt.figure(figsize=(10, 6))
plt.ylim([0, 1])
plt.plot(ac_sft_4_baseline, 'b', label='alpha=0.1')
plt.plot(ac_sft_4, 'lightskyblue', label='alpha=0.1 without baseline')
plt.plot(ac_sft_1_baseline, 'r', label='alpha=0.4')
plt.plot(ac_sft_1, 'lightcoral', label='alpha=0.4 without baseline')
plt.legend()
plt.show()
```

## Conclusion
This repository demonstrates the implementation and performance comparison of different Multi-Armed Bandit strategies. The results help understand the trade-offs between exploration and exploitation, as well as the impact of initialization and preferences on action selection.

## Installation
To run the code, ensure you have the following libraries installed:

```sh
pip install numpy matplotlib tqdm
```

## Usage
Clone this repository and run the provided scripts to visualize the results:

```sh
git clone https://github.com/rahulvyasm/multi-armed-bandit-rl.git
cd multi-armed-bandit-rl
python multi-bandit-problem.py
```

## License
This project is licensed under the MIT License.

---
