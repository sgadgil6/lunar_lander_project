import matplotlib.pyplot as plt


rewards_64 = []
episodes_64 = []
count = 0
rewards_128 = []
episodes_128 = []
rewards_128_engine_failure = []
episodes_128_engine_failure = []


with open('./results/evaluate.txt') as f:
    for line in f:
        line = line.split('=')
        if len(line) != 1:
        #print(float(line[3].strip()))
            rewards_64.append(float(line[3].strip()))
            episodes_64.append(count)
            count += 1
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
ax.set_title("Comparison Under Engine Failure")
plt.plot(episodes_64, rewards_64, 'r', label='Vanilla DQN')
plt.plot(episodes_128, rewards_128, 'g', label='Trained DQN without engine failure ')
plt.plot(episodes_128_engine_failure, rewards_128_engine_failure, 'b', label='Trained DQN with engine failure')
ax.set_xlabel("Iterations")
ax.set_ylabel("Average Reward")
ax.legend(loc='best')
plt.savefig('dqn_plot_engine_failure.jpg')
plt.show()