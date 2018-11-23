import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
from tqdm import *
import os
import numpy as np
from PIL import Image
import visdom
from src.model.autoencoder import autoencoder
from random import randint, random

def to_img(x):
	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	return x

def train(load_model, learning_rate, num_epochs):
	""" Trains the autoencoder.

	Keyword arguments:
	load_model -- load previous model
	learning_rate -- learning rate of the algorithm
	num_epochs -- training epochs (optional)
	"""

	if not os.path.exists('./doom_img'):
		os.mkdir('./doom_img')	

	batch_size = 128
	code_size = 1024
	linear_input = 2304
	linear_output = 5760
	vis = visdom.Visdom()     

	img_transform = transforms.Compose([
		transforms.Resize((60,108), Image.ANTIALIAS),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	dataset = datasets.ImageFolder(root='./training_set/', transform=img_transform)

	dataset_length = len(dataset)

	#Training
	n_training_samples = (dataset_length / 5) * 3
	train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

	#Validation
	n_val_samples = (dataset_length / 5)
	val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
	print(n_training_samples, n_training_samples + n_val_samples)

	#Test
	n_test_samples = (dataset_length / 5)
	test_sampler = SubsetRandomSampler(np.arange(n_training_samples + n_val_samples, n_training_samples + n_val_samples + n_test_samples, dtype=np.int64))
	print(n_training_samples + n_val_samples, n_training_samples + n_val_samples + n_test_samples)

	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
											   sampler=train_sampler, num_workers=2, drop_last=True)
	validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
													sampler=val_sampler, num_workers=2, drop_last=True)
	test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
													sampler=test_sampler, num_workers=2, drop_last=True)

	train_loss_vector = []
	val_loss_vector = []
	epoch_vector = []

	if load_model:
		model = torch.load('./src/model/autoencoder.pth')
	else:
		model = autoencoder(linear_input, linear_output, code_size).cuda()

	criterion = nn.MSELoss()
	#criterion = nn.BCELoss()
	m = nn.Sigmoid()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):    
		print('Training')
		for data in tqdm(train_loader):
			img, _ = data
			img = Variable(img).cuda()
			# ===================forward=====================
			output, _ = model(img)
			loss = criterion(output, img)
			# ===================backward====================
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		# ===================log========================
		print('epoch [{}/{}], loss:{:.4f}'
			  .format(epoch+1, num_epochs, loss.item()))
		
		if epoch % 1 == 0:
			pic = to_img(output.cpu().data)
			save_image(pic, './doom_img/image_{}.png'.format(epoch))

		train_loss_vector.append(loss.item())
			
		print('Validation')
		for data in tqdm(validation_loader):
			img, _ = data
			img = Variable(img).cuda()
			# ===================forward=====================
			output, _ = model(img)
			val_loss = criterion(output, img)
			# ===================backward====================
			optimizer.zero_grad()
			val_loss.backward()
			optimizer.step()
		# ===================log========================
		print('Validation loss:{:.4f}'.format(val_loss.item()))

		val_loss_vector.append(val_loss.item())
		epoch_vector.append(epoch)
		validation = dict(x=epoch_vector, y=val_loss_vector, mode="markers+lines", 
					type='custom', marker={'color': 'red', 'symbol': 104, 'size': "10"})
		train = dict(x=epoch_vector, y=train_loss_vector, mode="markers+lines", 
					type='custom', marker={'color': 'blue', 'symbol': 104, 'size': "10"})
		layout = dict(title="Loss function", xaxis={'title': 'epochs'}, yaxis={'title': 'loss'})

		vis._send({'data': [validation, train], 'layout': layout, 'win': 'aelosswin'}) 
		torch.save(model, './src/model/autoencoder.pth')
		
	print('Testing')
	for data in tqdm(test_loader):
		img, _ = data
		img = Variable(img).cuda()
		# ===================forward=====================
		output, _ = model(img)
		test_loss = criterion(output, img)
		# ===================backward====================
		optimizer.zero_grad()
		test_loss.backward()
		optimizer.step()
	# ===================log========================
	print('Testing loss:{:.4f}'.format(test_loss.item()))
	pic = to_img(output.cpu().data)
	save_image(pic, './doom_img/testing_{}.png'.format(epoch))

	torch.save(model, './src/model/autoencoder.pth')


if __name__ == "__main__":  
	train()