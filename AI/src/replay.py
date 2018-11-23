from __future__ import print_function
from random import choice, randint
from vizdoom import *
import os
from PIL import Image
import numpy as np

def screenshot(state, number):
	""" Save an state as image

	Keyword arguments: 
	state -- state or frame to save
	number -- frame number
	"""

	image_name = './training_set/train/frame_{}.jpg'.format(number)
	state = np.ascontiguousarray(state.transpose(1,2,0))
	img = Image.fromarray(state, 'RGB')
	img.save(image_name)

def recording(replay_file, config_file):
	""" Start a recording

	Keyword arguments: 
	replay_file -- path to save the replay
	config_file -- config .cfg file to load the map and configuration
	"""

	print("Initializing doom...")
	game = DoomGame()
	game.load_config("../scenarios/" + config_file)
	game.set_screen_resolution(ScreenResolution.RES_640X480)
	game.set_render_hud(True)
	game.set_mode(Mode.SPECTATOR)
	game.init()
	print("Doom initialized.")

	path = './replay/'

	actions = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]]
	
	game.new_episode(path + replay_file)
	while not game.is_episode_finished():
		s = game.get_state()
		game.advance_action()

		r = game.get_last_reward()

	print("Episode finished.")
	print("total reward:", game.get_total_reward())
	print("************************")

	game.close()

def replay(replay_file):
	""" Watch replay and adds all frames to the training set

	Keyword arguments: 
	replay_file -- file name of replay to watch
	"""

	print("Initializing doom...")
	game = DoomGame()
	game.load_config("../scenarios/replay.cfg")
	game.set_screen_resolution(ScreenResolution.RES_400X225)
	game.set_render_hud(True)
	game.set_mode(Mode.SPECTATOR)
	game.init()
	print("Doom initialized.")

	game.replay_episode('./replay/' + replay_file)
	frame_number = 0

	counter = len([name for name in os.listdir('./training_set/train/')])

	if counter > 0:
		frame_number = frame_number + counter

	while not game.is_episode_finished():
		s = game.get_state()
		game.advance_action()
		screenshot(s.screen_buffer, frame_number)
		frame_number = frame_number + 1

		r = game.get_last_reward()

	print("Episode finished.")
	print("total reward:", game.get_total_reward())
	print("************************")

	game.close()
