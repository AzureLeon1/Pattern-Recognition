import gym
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['figure.dpi'] = 300
# plt.rcParams.update({'font.size': 3})

environment = gym.make("FrozenLake-v1", is_slippery=False)
# Evaluation
def evaluate(episodes_eval=80):
    nb_success = 0
    cnt_steps = []

    for _ in range(episodes_eval):
        state = environment.reset()
        done = False

        cnt_step = 0

        history = []

        # Until the agent gets stuck or reaches the goal, keep training it
        while not done:
            # Choose the action with the highest value in the current state
            action = np.argmax(qtable[state])
            history.append(action)

            # Implement this action and move the agent in the desired direction
            new_state, reward, done, info = environment.step(action)

            # Update our current state
            state = new_state

            # When we get a reward, it means we solved the game
            nb_success += reward
            cnt_step += 1
        if reward == 1:
            cnt_steps.append(cnt_step)
            # print(history)
    # Let's check our success rate!
    # print(f"Success rate = {nb_success / episodes_eval * 100}%")
    return nb_success / episodes_eval * 100, np.mean(cnt_steps) if len(cnt_steps)>0 else np.nan


qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hyper-parameters for training
episodes = 1000  # Total number of episodes
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease

# List of outcomes to plot
outcomes = []
success_rates = []
avg_steps = []


print('Q-table before training:')
print(qtable)

# Training
for id in range(episodes):
    print(id)
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

        rnd = np.random.random()
        if rnd < epsilon:
            next_action = environment.action_space.sample()
        # Else, take the action with the highest value in the current state
        else:
            next_action = np.argmax(qtable[new_state])

        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * qtable[new_state, next_action] - qtable[state, action])

        # Update our current state
        state = new_state

        # If we have a reward, it means that our outcome is a success
        if reward:
            outcomes[-1] = "Success"

        if done:
            success_rate, avg_step = evaluate()
            success_rates.append(success_rate)
            if not np.isnan(avg_step):
                avg_steps.append(avg_step)

    # Update epsilon
    epsilon = max(epsilon - epsilon_decay, 0)

print()
print('===========================================')
print('Q-table after training:')
print(qtable)

print(success_rates)
print(avg_steps)


plt.xlabel("#Episodes")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=0.6)
plt.show()

# evaluate(episodes_eval=80)

plt.xlabel("#Episodes")
plt.ylabel("Success Rate")
plt.plot(range(len(success_rates)), success_rates)
plt.show()


plt.xlabel("#Episodes")
plt.ylabel("Average Steps")
plt.plot(range(len(avg_steps)), avg_steps)
plt.show()
