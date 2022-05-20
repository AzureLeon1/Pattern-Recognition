import gym
import numpy as np
import matplotlib.pyplot as plt


environment = gym.make("FrozenLake-v1", is_slippery=False)
# Evaluation
def evaluate(episodes_eval=80):
    print('===========================================')
    print('开始测试：')
    nb_success = 0
    cnt_steps = []

    for _ in range(episodes_eval):
        state = environment.reset()
        done = False
        cnt_step = 0
        while not done:
            action = np.argmax(qtable[state])
            new_state, reward, done, info = environment.step(action)
            state = new_state
            nb_success += reward
            cnt_step += 1
        if reward == 1:
            cnt_steps.append(cnt_step)
    print(f"成功率 = {nb_success / episodes_eval * 100}%")
    print(f"平均成功路径长度 = {np.mean(cnt_steps) if len(cnt_steps) > 0 else 0}")
    return nb_success / episodes_eval * 100, np.mean(cnt_steps) if len(cnt_steps)>0 else np.nan


qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# 超参数
episodes = 1000  # Total number of episodes
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease

# 训练时每个 episode 的成功情况，用于可视化
outcomes = []

print('训练前的 Q-table：')
print(qtable)

# Training
for _ in range(episodes):
    state = environment.reset()
    done = False

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Generate a random number between 0 and 1
        rnd = np.random.random()

        # If random number < epsilon, take a random action
        if rnd < epsilon:
            action = environment.action_space.sample()
        # Else, take the action with the highest value in the current state
        else:
            action = np.argmax(qtable[state])

        # Implement this action and move the agent in the desired direction
        new_state, reward, done, info = environment.step(action)

        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # Update our current state
        state = new_state

        # If we have a reward, it means that our outcome is a success
        if reward:
            outcomes[-1] = "Success"

    # Update epsilon
    epsilon = max(epsilon - epsilon_decay, 0)

print()
print('===========================================')
print('训练后的 Q-table：')
print(qtable)

plt.xlabel("#Episodes")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=0.6)
plt.show()

evaluate(episodes_eval=80)
