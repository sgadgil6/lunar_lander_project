import matplotlib.pyplot as plt


rewards_noisy_obs_train= []
episodes_noisy_obs_train = []
count = 0
rewards_noisy_obs = []
episodes_noisy_obs = []

with open('./results/dqn_noisy_obs.txt') as f:
    for line in f:
        line = line.split('=')
        if len(line) != 1:
        #print(float(line[3].strip()))
            rewards_noisy_obs.append(float(line[3].strip()))
            episodes_noisy_obs.append(count)
            count += 1
count=0
with open('./results/dqn_noisy_obs_train.txt') as f:
    for line in f:
        line = line.split('=')
        #print(float(line[3].strip()))
        rewards_noisy_obs_train.append(float(line[3].strip()))
        episodes_noisy_obs_train.append(count)
        count += 1



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Comparison Under Noisy Observation")
plt.plot(episodes_noisy_obs_train, rewards_noisy_obs_train, 'g', label='Trained DQN')
plt.plot(episodes_noisy_obs, rewards_noisy_obs, 'b', label='Vanilla DQN')
ax.set_xlabel("Iterations")
ax.set_ylabel("Average Reward")
ax.legend(loc='best')
plt.savefig('dqn_plot_noisy_obs.jpg')
plt.show()