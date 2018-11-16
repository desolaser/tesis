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
from src.model.autoencoder import autoencoder

if not os.path.exists('./doom_img'):
	os.mkdir('./doom_img')

def to_img(x):
	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	return x

num_epochs = 40
batch_size = 128
learning_rate = 1e-3
code_size = 4096
linear_input = 9216
linear_output = 12960
load_model = False

img_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
#2269
dataset = datasets.ImageFolder(root='./training_set_min/', transform=img_transform)

#Training
n_training_samples = 1361
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

#Validation
n_val_samples = 454
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

#Test
n_test_samples = 454
test_sampler = SubsetRandomSampler(np.arange(n_training_samples + n_val_samples, n_training_samples + n_val_samples + n_test_samples, dtype=np.int64))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
										   sampler=train_sampler, num_workers=2)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
												sampler=val_sampler, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
												sampler=test_sampler, num_workers=2)

if load_model:
	model = torch.load('./src/model/autoencoder.pth')
else:
	model = autoencoder(linear_input, linear_output, code_size).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
							 weight_decay=1e-5)

if __name__ == "__main__":  
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
		
		if epoch % 10 == 0:
			pic = to_img(output.cpu().data)
			save_image(pic, './doom_img/image_{}.png'.format(epoch))
			
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

	torch.save(model, './src/model/autoencoder.pth')
