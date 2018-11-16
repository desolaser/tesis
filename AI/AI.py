#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from tqdm import trange
import click
import visdom
from functools import partial
import pickle
from PIL import Image
import numpy as np
from src.replay_memory import ReplayMemory
from src.model.autoencoder import autoencoder
from src.model.dqn import dqnNet as Net
from src.model.qnet import qNet
from src.model.drqn import drqnNet
import src.ae_trainer as ae_trainer
import src.tools as tools
import os

class Config(object):
	"""Defines the configuration and global variables"""

	def __init__(self):
		# Q-learning settings
		self.epochs = 5
		self.learning_steps_per_epoch = 2000
		self.replay_memory_size = 10000

		# NN learning settings
		self.batch_size = 32

		# Training regime
		self.test_episodes_per_epoch = 100        
		self.frame_repeat = 4

		# Other parameters
		self.resolution = (3, 60, 108)
		#self.resolution = (3, 72, 96)
		self.episodes_to_watch = 10

		self.criterion = nn.MSELoss()

		# Configuration file path
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.vis = visdom.Visdom()        
		self.loss_vector = []  
		self.reward_vector = []
		self.epoch_vector = []
		self.actual_epoch_loss_vector = []
		self.sw = False

		self.code_size = 4096

pass_config = click.make_pass_decorator(Config, ensure=True)

@click.group()
def cli():
	pass

@cli.command()
@click.option('--load_model', is_flag=True, help='Load previous model?')
@click.option('--learning_rate', type=float, default='1e-3', help='Autoencoder learning rate')
def ae_train(load_model, learning_rate): 
	ae_trainer.train(load_model, learning_rate)

@pass_config
def preprocess(config, img): 
	img = skimage.transform.resize(img, config.resolution)
	img = img.astype(np.float32)	
	return img

@pass_config
def learn(config, s1, target_q):		
	s1 = torch.from_numpy(s1)
	target_q = torch.from_numpy(target_q)
	if torch.cuda.is_available():
		s1, target_q = s1.to(config.device), target_q.to(config.device)

	s1, target_q = Variable(s1), Variable(target_q)

	if config.daqn:
		_, code = config.autoencoder(s1)
		output = config.model(code)	
	elif config.drqn:						
		config.cx = Variable(config.cx.data)
		config.hx = Variable(config.hx.data)
		inputs = s1, (config.hx, config.cx)
		output, (config.hx, config.cx) = config.model(inputs)
	else:
		output = config.model(s1)

	loss = config.criterion(output, target_q)
	config.optimizer.zero_grad()
	loss.backward()
	config.optimizer.step()
	return loss

def to_img(x):
	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	return x

@pass_config
def get_q_values(config, state):
	state = torch.from_numpy(state)
	if torch.cuda.is_available():
		state = state.to(config.device)

	state = Variable(state)

	if config.daqn:
		img_output, code = config.autoencoder(state)	
		#img_output = to_img(img_output.cpu().data)
		#save_image(img_output, './auto_img/image_{}.png'.format(randint(0,100000000000000000)))
		output = config.model(code)
	elif config.drqn:						
		config.cx = Variable(config.cx.data)
		config.hx = Variable(config.hx.data)
		inputs = state, (config.hx, config.cx)
		output, (config.hx, config.cx) = config.model(inputs)
	else:
		output = config.model(state)

	return output

def get_best_action(state):
	q = get_q_values(state)
	m, index = torch.max(q, 1)
	action = index.cpu().data.numpy()[0]
	return action

@pass_config
def learn_from_memory(config):
	""" Learns from a transition, that comes from the replay memory.
	s2 is ignored if isTerminal equals true """

	if config.memory.size > config.batch_size:
		s1, a, s2, isterminal, r = config.memory.get_sample(config.batch_size)    
		q = get_q_values(s2).cpu().data.numpy() 
		q2 = np.max(q, axis=1)
		target_q = get_q_values(s1).cpu().data.numpy()
		target_q[np.arange(target_q.shape[0]), a] = r + config.discount_factor * (1 - isterminal) * q2

		loss = learn(s1, target_q)
		config.actual_epoch_loss_vector.append(loss.cpu().detach().numpy())

@pass_config
def perform_learning_step(config, epoch):
	""" Makes an action according to eps-greedy policy, observes the result
	(next state, reward) and learns from the transition"""

	def exploration_rate(config, epoch):
		""" Define exploration rate change over time"""
		start_eps = 1.0
		end_eps = 0.1
		const_eps_epochs = 0.1 * config.epochs  
		eps_decay_epochs = 0.6 * config.epochs  

		if epoch < const_eps_epochs:
			return start_eps
		elif epoch < eps_decay_epochs:
			return start_eps - (epoch - const_eps_epochs) / \
							   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
		else:
			return end_eps
	
	s1 = preprocess(config.game.get_state().screen_buffer)
	eps = exploration_rate(config, epoch)

	if random() <= eps:
		a = randint(0, len(config.actions) - 1)
	else:		
		s1 = s1.reshape([1, config.resolution[0], config.resolution[1], config.resolution[2]])
		a = get_best_action(s1)

	reward = config.game.make_action(config.actions[a], config.frame_repeat)
	isterminal = config.game.is_episode_finished()
	s2 = preprocess(config.game.get_state().screen_buffer) if not isterminal else None
	config.memory.add_transition(s1, a, s2, isterminal, reward)

	learn_from_memory()

def initialize_vizdoom(config_path):
	""" Initialize vizdoom game class an set configuration"""

	print("Initializing doom...")
	game = DoomGame()
	game.load_config(config_path)
	game.set_window_visible(False)
	game.set_mode(Mode.PLAYER)
	game.set_screen_format(ScreenFormat.CRCGCB)
	game.set_screen_resolution(ScreenResolution.RES_400X225)
	game.init()
	print("Doom initialized.")
	return game

def screenshot(game):
	image_name = './training_set/train/image_{}.jpg'.format(randint(0,100000000000000000)) 
	image_array = game.get_state().screen_buffer
	image_array = np.ascontiguousarray(image_array.transpose(1,2,0))
	img = Image.fromarray(image_array, 'RGB')
	img.save(image_name)     

@cli.command()
@click.option('--learning_rate', type=float, default='0.00025', help='Learning rate of the neuronal network')
@click.option('--discount_factor', type=float, default='0.99', help='Discount factor of the Q-learning algorithm')
@click.option('--config_path', default='../scenarios/basic.cfg', help='Config file .cfg')
@click.option('--model_to_load', help='Network model to load')
@click.option('--skip_learning', is_flag=True, help='Skip learning?')
@click.option('--skip_watching', is_flag=True, help='Skip watching the agent play?')
@click.option('--image_capture', is_flag=True, help='Capture images while watching the agent?')
@click.option('--daqn', is_flag=True, help='Use DAQN model?')
@click.option('--drqn', is_flag=True, help='Use DRQN model?')
@pass_config
def train(config, learning_rate, discount_factor, config_path, 
	model_to_load, skip_learning, skip_watching, image_capture, daqn, drqn):
	""" Train and watch the model train"""

	config.learning_rate = learning_rate
	config.discount_factor = discount_factor
	config.config_path = '../scenarios/'+config_path
	config.model_to_load = model_to_load
	config.daqn = daqn
	config.drqn = drqn

	config.game = initialize_vizdoom(config.config_path)

	n = config.game.get_available_buttons_size()
	config.actions = [list(a) for a in it.product([0, 1], repeat=n)]

	config.memory = ReplayMemory(config.resolution, config.replay_memory_size)  
	
	if config.daqn:			
		config.autoencoder = torch.load('./src/model/autoencoder.pth')

	if model_to_load:
		print("Loading model from: ", config.model_to_load)
		""" This code doesn't work in pytorch 1.0.0, delete it if you use that version. """

		pickle.load = partial(pickle.load, encoding="latin1")
		pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
		config.model = torch.load(config.model_to_load, map_location=lambda storage, loc: storage, pickle_module=pickle)

		""" The code works in pytorch 1.0.0, if you use this version then uncomment the code below. """
		# config.model = torch.load(config.model_to_load)
	else:
		if config.daqn:
			config.model = qNet(len(config.actions), config.code_size)	
		elif config.drqn:
			linear_input = tools.layerCalculator(config.resolution[1], config.resolution[2])
			config.model = drqnNet(len(config.actions), linear_input)
		else:
			linear_input = tools.layerCalculator(config.resolution[1], config.resolution[2])
			config.model = Net(len(config.actions), linear_input)

	if torch.cuda.is_available():
		config.model.to(config.device)	

	config.optimizer = torch.optim.SGD(config.model.parameters(), config.learning_rate)    

	print("Starting the training!")
	print("Learning rate: ", config.learning_rate)
	print("Discount factor: ", config.discount_factor)
	print("Epochs: ", config.epochs)
	print("Learning steps per epoch: ", config.learning_steps_per_epoch)
	print("Batch size: ", config.batch_size)
	
	time_start = time()
	if not skip_learning:
		for epoch in range(config.epochs):
			print("\nEpoch %d\n-------" % (epoch + 1))
			train_episodes_finished = 0
			train_scores = []           

			print("Training...")
			config.game.new_episode()
			if config.drqn:						
				config.cx = Variable(torch.zeros(config.batch_size, len(config.actions)).cuda())
				config.hx = Variable(torch.zeros(config.batch_size, len(config.actions)).cuda())

			for learning_step in trange(config.learning_steps_per_epoch, leave=False):
				perform_learning_step(epoch)
				if config.game.is_episode_finished():
					score = config.game.get_total_reward()
					train_scores.append(score)
					config.game.new_episode()					
					if config.drqn:						
						config.cx = Variable(torch.zeros(config.batch_size, len(config.actions)).cuda())
						config.hx = Variable(torch.zeros(config.batch_size, len(config.actions)).cuda())
					train_episodes_finished += 1

			print("%d training episodes played." % train_episodes_finished)

			train_scores = np.array(train_scores)

			print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
				  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

			# graph loss
			average_loss = sum(config.actual_epoch_loss_vector) / len(config.actual_epoch_loss_vector)
			config.loss_vector.append(average_loss)
			config.epoch_vector.append(epoch)

			trace = dict(x=config.epoch_vector, y=config.loss_vector, mode="markers+lines", 
					type='custom', marker={'color': 'red', 'symbol': 104, 'size': "10"})
			layout = dict(title="Loss function", xaxis={'title': 'epochs'}, yaxis={'title': 'loss'})

			config.vis._send({'data': [trace], 'layout': layout, 'win': 'losswin'})         
			config.actual_epoch_loss_vector = []

			print("\nTesting...")
			test_episode = []
			test_scores = []
			for test_episode in trange(config.test_episodes_per_epoch, leave=False):
				config.game.new_episode()				
				if config.drqn:						
					config.cx = Variable(torch.zeros(1, len(config.actions)).cuda())
					config.hx = Variable(torch.zeros(1, len(config.actions)).cuda())
					
				while not config.game.is_episode_finished():
					state = preprocess(config.game.get_state().screen_buffer)
					state = state.reshape([1, config.resolution[0], config.resolution[1], config.resolution[2]])
					best_action_index = get_best_action(state)

					config.game.make_action(config.actions[best_action_index], config.frame_repeat)
				r = config.game.get_total_reward()
				test_scores.append(r)

			test_scores = np.array(test_scores)
			print("Results: mean: %.1f +/- %.1f," % (
				test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
				  "max: %.1f" % test_scores.max())

			save_file = "./nets/model_{}.pth".format(epoch)
			print("Saving the network weigths to:", save_file)
			torch.save(config.model, save_file)         

			print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))          

			# graph reward
			config.reward_vector.append(test_scores.mean())

			trace = dict(x=config.epoch_vector, y=config.reward_vector, mode="markers+lines", 
					type='custom', marker={'color': 'red', 'symbol': 104, 'size': "10"})
			layout = dict(title="Testing reward graph", xaxis={'title': 'epochs'}, yaxis={'title': 'reward'})           

			config.vis._send({'data': [trace], 'layout': layout, 'win': 'rewardwin'})
				

	config.game.close()

	if not skip_watching:
		print("======================================")
		print("Training finished. It's time to watch!")

		config.game.set_window_visible(True)
		config.game.set_mode(Mode.ASYNC_PLAYER)
		config.game.init()

		score_array = []
		score = 0

		sc_counter = 0
		
		for _ in range(config.episodes_to_watch):
			config.game.new_episode()			
			if config.drqn:						
				config.cx = Variable(torch.zeros(1, len(config.actions)).cuda())
				config.hx = Variable(torch.zeros(1, len(config.actions)).cuda())

			while not config.game.is_episode_finished():

				if image_capture and sc_counter >= 2:
					screenshot(config.game)		
					sc_counter = 0		

				if image_capture:	
					sc_counter = sc_counter + 1

				state = preprocess(config.game.get_state().screen_buffer)
				state = state.reshape([1, config.resolution[0], config.resolution[1], config.resolution[2]])
				best_action_index = get_best_action(state)
				config.game.set_action(config.actions[best_action_index])
				for _ in range(config.frame_repeat):
					config.game.advance_action()

			sleep(1.0)
			score = config.game.get_total_reward()
			print("Total score: ", score)
			score_array.append(score)

		average_score = sum(score_array) / 10
		print(average_score)
