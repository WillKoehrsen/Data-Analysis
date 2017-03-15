import random
import numpy as np 
import gym
from collections import Counter
from statistics import mean, median
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LR = 1e-3 # learning rate
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 100
initial_games = 50000


def some_random_games_first():
	
	scores = []
	# each of these is a game
	for game in range(5):
		score = 0 
		env.reset()
		# this is each frame, up to the goal number of frames
		for frame in range(goal_steps):
			# env.render displays the environment. It will make the script run slower, but provides a visualization of the game.
			# env.render()

			#Next, create a sample action in any environment. In the case of CartPole, the action is either a 0 or 1 (left or right).
			action = env.action_space.sample()

			# This executes the environment with an action, and returns the observation of the environment, the reward (score), if the env is over, and additional info
			observation, reward, done, info = env.step(action)
			score += reward

			if done:
				scores.append(score) 
				break
				
	return scores
# print(some_random_games_first())

def initial_population():
	# [OBS, MOVES]
	training_data = []
	# all scores:
	scores = []
	# only add the scores that meet the threshold to accepted_scores
	accepted_scores = []
	# iterate through the number of games we want
	for game in range(initial_games):
		score = 0
		# moves specifically from this environment:
		game_memory = []
		#previous observation from the previous frame, each frame returns an observation
		prev_observation = []
		# for each frame in the goal steps
		for frame in range(goal_steps):
			# choose a random action
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)
			
			# take action based on previous observation, action will be tied to previous observation
			# In actual games, we will take an action based on the previous observation. 
			# The computer learns what sequences of previous observations and actions led to a score above the requirement.
			# The neural network model predicts the action most likely to increase the scores based on the training and the previous observation.
			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])
			prev_observation = observation
			score += reward
			if done: break

		# IF the score is above the threshold, save every move made. 
		# This is how the program actually learns. The script is only reinforcing the score, not how the score was reached. 
		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				# converting to one-hot
				if data[1] == 1: # if action was 1
					output = [0, 1]
				elif data[1] == 0:
					output = [1, 0] 

				# tying observation to the action
				training_data.append([data[0], output])

		env.reset()
		scores.append(score)
	print('Average accepted score: %0.2f' % mean(accepted_scores))
	print('Median accepted score:', median(accepted_scores))
	print(Counter(accepted_scores))
	return training_data

# mathematical model of the neural network
def neural_network_model(input_size):

	network = input_data(shape=[None, input_size, 1], name='input')

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	# output layer of the network
	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam',learning_rate=LR, loss='categorical_crossentropy', name='targets')
	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

def train_model(training_data, model=False):

	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]),1)
	y = [i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size=len(X[0]))

	model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')

	return model

training_data = initial_population()
model = train_model(training_data)	

scores = []
choices = []

for game in range(1):
	env.reset()
	score = 0
	game_memory = []
	prev_observation = []
	for frame in range(goal_steps):
		env.render()

		if len(prev_observation) == 0:
			action = random.randrange(0,2)
		else:
			action = np.argmax(model.predict(prev_observation.reshape(-1, len(prev_observation),1))[0])
			# print(model.predict(prev_observation.reshape(-1, len(prev_observation),1))[0])
		choices.append(action)

		new_observation, reward, done, info = env.step(action)
		prev_observation = new_observation
		game_memory.append([new_observation, action])
		score += reward
		if done: break

	scores.append(score)

print('Average score:', sum(scores)/len(scores))
print('Choice 0: {} Choice 1: {}'.format(choices.count(0)/len(choices), choices.count(1)/len(choices)))


