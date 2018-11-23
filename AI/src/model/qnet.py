import torch.nn as nn
import torch.nn.functional as F
import math

class qNet(nn.Module):
	def __init__(self, available_actions_count, input_size):
		super(qNet, self).__init__() 
		self.input_size = input_size
		self.fc1 = nn.Linear(input_size, 512)
		self.fc2 = nn.Linear(512, available_actions_count)

	def forward(self, x):	
		x = F.relu(self.fc1(x))
		return self.fc2(x)

