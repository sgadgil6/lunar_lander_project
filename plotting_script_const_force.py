import matplotlib.pyplot as plt
import numpy as np

rewards_00 = []
episodes_00 = []
count = 0
rewards_01 = []
episodes_01 = []
rewards_10 = []
episodes_10 = []
rewards_11 = []
episodes_11 = []
rewards_medium = []
episodes_medium = []
rewards_high = []
episodes_high = []




with open('./results/dqn_const_force_00.txt') as f:
    for line in f:
        line = line.split('=')
        if len(line) != 1:
        #print(float(line[3].strip()))
            rewards_00.append(float(line[3].strip()))
            episodes_00.append(count)
            count += 1
count=0
with open('./results/dqn_const_force_01.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_01.append(float(line[3].strip()))
        episodes_01.append(count)
        count += 1

count=0
with open('./results/dqn_const_force_10.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_10.append(float(line[3].strip()))
        episodes_10.append(count)
        count += 1

count=0
with open('./results/dqn_const_force_11.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_11.append(float(line[3].strip()))
        episodes_11.append(count)
        count += 1

count=0
with open('./results/dqn_const_force_medium.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_medium.append(float(line[3].strip()))
        episodes_medium.append(count)
        count += 1

count=0
with open('./results/dqn_const_force_high.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_high.append(float(line[3].strip()))
        episodes_high.append(count)
        count += 1


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Comparison With Random Force")
plt.plot(episodes_00, rewards_00, 'r', label='regular_00')
plt.plot([0, 400], [np.mean(rewards_00), np.mean(rewards_00)], '--',color='r')
plt.plot(episodes_01, rewards_01, 'b', label='regular_01 ')
plt.plot([0, 400], [np.mean(rewards_01), np.mean(rewards_01)], '--',color='b')
plt.plot(episodes_10, rewards_10, 'y', label='regular_10')
plt.plot([0, 400], [np.mean(rewards_10), np.mean(rewards_10)], '--',color='y')
plt.plot(episodes_11, rewards_11, 'k', label='regular_11')
plt.plot([0, 400], [np.mean(rewards_11), np.mean(rewards_11)], '--',color='k')
plt.plot(episodes_medium, rewards_medium, 'c', label='regular_medium')
plt.plot([0, 400], [np.mean(rewards_medium), np.mean(rewards_medium)], '--',color='c')
plt.plot(episodes_high, rewards_high, 'g', label='regular_high')
plt.plot([0, 400], [np.mean(rewards_high), np.mean(rewards_high)], '--',color='g')
plt.plot()
ax.set_xlabel("Iterations")
ax.set_ylabel("Average Reward")
ax.legend()
#ax.set_ylim([0, 500])
plt.savefig('dqn_plot_const_force.jpg')
plt.show()