import numpy as np
import keras
from keras.models import load_model
import lunar_lander as lander


def get_obs(true_loc):
    tX = true_loc[0]
    tY = true_loc[1]

    nX = np.random.normal(loc=tX, scale=0.1)

    return (nX, tY)

if __name__ == '__main__':
    env = lander.LunarLander()
    model = load_model('.\saved_models\model_300_episodes_128.h5')
    scores = []
    for i in range(400):
        finished = False
        score = 0
        #np.random.seed(0)
        state = env.reset()
        iterations = 0
        while True:
            state = np.reshape(state, (1, 8))

            obs = get_obs((state[0][0], state[0][1]))
            state[0][0] = obs[0]
            state[0][1] = obs[1]
            action_values = model.predict(state)
            action = np.argmax(action_values[0])
            #still_open = env.render()
            env.render()
            #if still_open == False:
            #    break
            next_state, reward, finished, metadata = env.step(action)
            next_state = np.reshape(next_state, (1, 8))
            score+=reward
            state = next_state
            if finished or iterations > 500:
                scores.append(score)
                break
            #if iterations == 300:
            #    finished = True
            #if finished:
            #    scores.append(score)
            #    print("Episode = {}, Score = {}, Avg_Score = {}".format(i, score, np.mean(scores[-100:])))
            #    break
            iterations+=1
        #scores.append(score)
        print("Episode = {}, Score = {}, Avg_Score = {}".format(i, score, np.mean(scores[-10:])))
    print("Mean Score={}".format(np.mean(scores)))

