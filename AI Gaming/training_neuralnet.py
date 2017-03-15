import random
from statistics import mean, median
from collections import Counter
import numpy as np 
import gym
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression 

LR = 1e-4
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 100 # only add to training data if score > score_requirement
initial_games = 10000

def some_random_games_first():
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action = env.action_space.sample()
			# observation: information about the game, reward: 1 or 0; done: is the game over; info: any additional information
			observation, reward, done, info = env.step(action)
			return observation, reward, done, info
			if done:
				break
			
# observation, reward, done, info = some_random_games_first()

def initial_population():
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])
			prev_observation = observation 
			score += reward
			if done: break

		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]

				training_data.append([data[0], output])

		env.reset()
		scores.append(score)
	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)

	print('Average accepted score: %0.3f' % mean(accepted_scores))
	print('Median accepted score:', median(accepted_scores))
	print(Counter(accepted_scores))

	return training_data

def neural_network_model(input_size):
	network = input_data(shape=[None, input_size, 1], name='input')
	
	network = fully_connected(network, 128, activation='relu') # rectified linear
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu') # rectified linear
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu') # rectified linear
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu') # rectified linear
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu') # rectified linear
	network = dropout(network, 0.8)

	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=LR,
						 loss='categorical_crossentropy', name='targets')
	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

def train_model(training_data, model=False):
	# training_data = observations, action taken: 0 or 1
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	y = [i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size = len(X[0]))

	model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500,
			  show_metric=True, run_id='openai')

	return model

training_data = initial_population()
model = train_model(training_data)

#model.save('openAICartPole.model')
#model.load('openAICartPole.model')

scores = []
choices = []

for each_game in range(10):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		# env.render()
		if len(prev_obs) == 0:
			action = random.randrange(0,2) # either a 0 or 1
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
		choices.append(action)

		new_observation, reward, done, info = env.step(action)
		prev_observation = new_observation
		game_memory.append([new_observation, action])
		score += reward
		if done: break
	scores.append(score)

print(scores)
print('Average Score', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
