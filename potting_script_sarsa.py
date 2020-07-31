import matplotlib.pyplot as plt
import numpy as np

rewards_eval = []
episodes_eval = []
count = 0
rewards_train = []
episodes_train = []
rewards_128_engine_failure = []
episodes_128_engine_failure = []


with open('./results/sarsa_engine_failure_eval.txt') as f:
    for line in f:
        line = line.split()
        #print(line)
        if line[0] != 'Overall':
             rewards_eval.append(float(line[4].strip()))
             episodes_eval.append(count)
             count += 10
count = 0
with open('./results/sarsa_engine_failure_training.txt') as f:
    for line in f:
        line = line.split()
        #print(line)
        rewards_train.append(float(line[4].strip()))
        episodes_train.append(count)
        count += 10

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Comparison Under Engine Failure")
plt.plot(episodes_train, rewards_train, 'orange', label='Trained Q')
plt.plot([0, 1000], [np.mean(rewards_train), np.mean(rewards_train)], '--',color='orange')
plt.plot(episodes_eval, rewards_eval, 'green', label='Vanilla Q')
plt.plot([0, 1000], [np.mean(rewards_eval), np.mean(rewards_eval)], '--',color='green')
ax.set_xlabel("Iterations")
ax.set_ylabel("Average Reward")
ax.legend(loc='best')
plt.savefig('sarsa_plot_engine_failure.jpg')
plt.show()
"""
count=0
with open('./results/dqn_128.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_128.append(float(line[3].strip()))
        episodes_128.append(count)
        count += 1

count=0
with open('./results/dqn_128_engine_failure.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_128_engine_failure.append(float(line[3].strip()))
        episodes_128_engine_failure.append(count)
        count += 1




fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Acquired Rewards")
plt.plot(episodes_64, rewards_64, 'r', label='Trained DQN')
plt.plot(episodes_128, rewards_128, 'g', label='DQN 128 without engine failure ')
plt.plot(episodes_128_engine_failure, rewards_128_engine_failure, 'b', label='DQN 128 with engine failure')
ax.set_xlabel("Iterations")
ax.set_ylabel("Average Reward")
ax.legend(loc='best')
plt.savefig('dqn_plot_engine_failure.jpg')
plt.show()
"""