import torch.nn as nn
import torch.nn.functional as F

class autoencoder(nn.Module):
    def __init__(self, linear_input, linear_output, code_size):
        super(autoencoder, self).__init__()
        self.linear_input = linear_input
        self.linear_output = linear_output
        self.code_size = code_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1), 
            nn.Conv2d(16, 32, 5, stride=2),  
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.linear_encoder = nn.Sequential(
            nn.Linear(linear_input, code_size),
            nn.ReLU(True)
        )

        self.linear_decoder = nn.Sequential(
            nn.Linear(code_size, linear_output),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=3, padding=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 6, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 9, stride=1),
            nn.Tanh()
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)       
        inputs = inputs.view(-1, self.linear_input)
        code = self.linear_encoder(inputs)
        output = self.linear_decoder(code)
        output = output.view(-1, 32, 10, 18)
        output = self.decoder(output)
        return output, code