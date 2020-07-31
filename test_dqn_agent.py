import numpy as np
import keras
from keras.models import load_model
import lunar_lander_const_force as lander

if __name__ == '__main__':
    env = lander.LunarLander()
    model = load_model('.\saved_models\dqn_const_force_model_350_episodes.h5')
    scores = []
    for i in range(400):
        finished = False
        score = 0
        #np.random.seed(0)
        state = env.reset()
        iterations = 0
        while iterations != 2000:
            state = np.reshape(state, (1, 8))
            action_values = model.predict(state)
            action = np.argmax(action_values[0])
            #still_open = env.render()
            env.render()
            #if still_open == False:
            #    break
            if np.random.random() > 0.2:
                next_state, reward, finished, metadata = env.step(action)
            else:
                next_state, reward, finished, metadata = env.step(0)
            #next_state, reward, finished, metadata = env.step(action)
            next_state = np.reshape(next_state, (1, 8))
            score+=reward
            state = next_state
            if finished:
                break
            #if iterations == 300:
            #    finished = True
            #if finished:
            #    scores.append(score)
            #    print("Episode = {}, Score = {}, Avg_Score = {}".format(i, score, np.mean(scores[-100:])))
            #    break
            iterations+=1
        scores.append(score)
        print("Episode = {}, Score = {}, Avg_Score = {}".format(i, score, np.mean(scores[-10:])))
    print("Mean Score={}".format(np.mean(scores)))

