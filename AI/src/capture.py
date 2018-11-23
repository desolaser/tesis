#!/usr/bin/env python

from vizdoom import *
from random import choice
from time import sleep
import os, os.path
import numpy as np
from PIL import Image
from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2grey
from tqdm import *

def initialize_vizdoom(config_path):	
	""" Initialize vizdoom game class an set configuratio."""

	print("Initializing doom...")
	game = DoomGame()
	game.load_config(config_path)
	game.set_window_visible(True)
	game.set_mode(Mode.PLAYER)
	game.set_screen_format(ScreenFormat.CRCGCB)
	game.set_screen_resolution(ScreenResolution.RES_400X225)
	game.init()
	print("Doom initialized.")
	return game

def image_capture(config_path, episodes, image_limit):
	""" Performs image capture for addition to the training set."""

	config_path = '../scenarios/'+config_path
	actions = [[True, False, False], [False, True, False], [False, False, True]]
	sleep_time = 1.0 / DEFAULT_TICRATE # = 0.028

	counter = len([name for name in os.listdir('./training_set/train/')])

	if image_limit != 0:
		image_limit = image_limit + counter

	game = initialize_vizdoom(config_path)
	for i in range(episodes):

		print("Episode #" + str(i + 1))
		game.new_episode()

		while not game.is_episode_finished():

			state = game.get_state()

			n = state.number
			vars = state.game_variables
			screen_buf = state.screen_buffer
			depth_buf = state.depth_buffer
			labels_buf = state.labels_buffer
			automap_buf = state.automap_buffer
			labels = state.labels

			r = game.make_action(choice(actions))

			print("Frame #" + str(counter))

			save_name = './training_set/train/image_{}.jpg'.format(counter)
			screen_buf = np.ascontiguousarray(screen_buf.transpose(1,2,0))
			img = Image.fromarray(screen_buf, 'RGB')
			img.save(save_name)

			counter = counter + 1

			if counter == image_limit:
				game.close()
				print("Image capture finished.")
				print("************************")
				return 0

			if sleep_time > 0:
				sleep(sleep_time)

		print("Image capture finished.")
		print("************************")

	game.close()
	#compare_images()

