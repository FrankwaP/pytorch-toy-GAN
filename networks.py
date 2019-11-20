# -*- coding: utf-8 -*-
"""
Toy model to learn how to mess with a GAN model
The training data are generated using a normal distribution, meaning:
    - The generator is trained to generate fake normal distributions from noise (uniform distribution).
    - The discriminator is trained to detect if normal distributions are real or fake.

Both distributions -- normal and noise/uniform -- are generated on the fly.
It's nothing special regarding the noise distribution.
But regarding the normal distributions, it has 2 advantages:
    - this simulate infinite data
    - this makes the code much shorter

We need "enough" values in each samples to detect if it's a normal distribution.
Imagine the question "is [-0.1, 2, 3] a normal distribution?" to understand why...

To facilitate -- actually make possible? -- the training of the discriminator, we sort its input values.
To do that, a sort layer is used as 1st layer of the discriminator.
"""
#%% Loading libraries

import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import  shapiro
# optimizer to play with
from torch.optim import Adam


if torch.cuda.is_available():
    print('Yes! CUDA is available!')
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#%% Functions to generate the data

def generate_real_data(batch_size, sample_size):
    mu, sig = 300., 1.
    return mu + sig*torch.randn((batch_size, sample_size)).to(device)

def generate_noise(batch_size, noise_size):
    return torch.rand((batch_size, noise_size)).to(device)


#%% Discriminator and Generator

class sort_layer(nn.Module):
    def forward(self, x):
        return x.sort(dim=1)[0]

def get_discriminator(n_input, n_hidden, n_output=1):
    return nn.Sequential(
        sort_layer(),
        # nn.BatchNorm1d(n_input),   <== it was very stupid since when training D there are batches with 100% real and batches with 100% fake
        nn.Linear(n_input, n_hidden), nn.LeakyReLU(),
        nn.Linear(n_hidden, n_output), nn.Sigmoid())

def get_generator(n_input, n_hidden, n_output):
    return nn.Sequential(
        nn.Linear(n_input, n_hidden), nn.Tanh(),
        nn.Linear(n_hidden, n_output))

#%% Training the GAN: batch level

batch_size = 10000
noise_size = 10
sample_size = 1000
n_epoch = 10000
record_step = 100

D = get_discriminator(sample_size, 2**6, 1)
D.cuda()
D_optim = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))


G = get_generator(noise_size, 2**6, sample_size)
G.cuda()
G_optim = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss = nn.BCELoss()


y_real = torch.ones((batch_size, 1)).to(device)
y_fake = 1-y_real


#%%
n_loop, n_D_loop, n_G_loop = 0, 0, 0
G_samples, G_loops, G_titles = [], [], []

G_samples, G_epochs, G_titles = [], [], []
for epoch in range(n_epoch):
    for _ in range(20):
        ###
        # "Update the discriminator by ascending its stochastic gradient"
        # fake data should NOT be predicted as real (negative loss for "ascending")
        z_fake = generate_noise(batch_size, noise_size)
        x_fake = G(z_fake) #.detach()
        y_fake_pred = D(x_fake)
        D_fake_loss = -loss(y_fake_pred, y_real)
        # real data should NOT be predicted as fake (negative loss for "ascending")
        x_real = generate_real_data(batch_size, sample_size)
        y_real_pred = D(x_real)
        D_real_loss = -loss(y_real_pred, y_fake)
        ## combining losses
        D_loss = 0.5*(D_real_loss + D_fake_loss)
        D.zero_grad()
        D_loss.backward()
        D_optim.step()
        ###
        print('Epoch [%d/%d], D_real_loss: %5.2f, D_fake_loss: %5.2f, D_loss: %5.2f, G_loss: %5.2f'
              % (epoch+1, n_epoch, D_real_loss.data, D_fake_loss.data, D_loss.data, 0))

    for _ in range(50):
        ###
        # "Update the generator by descending its stochastic gradient"
        # fake data should be predicted as real
        z_fake = generate_noise(batch_size, noise_size)
        x_fake = G(z_fake)
        y_pred = D(x_fake)
        G_loss = loss(y_pred, y_real)
        #
        G.zero_grad()
        G_loss.backward()
        G_optim.step()
        ###
        print('Epoch [%d/%d], D_real_loss: %5.2f, D_fake_loss: %5.2f, D_loss: %5.2f, G_loss: %5.2f'
              % (epoch+1, n_epoch, D_real_loss.data, D_fake_loss.data, D_loss.data, G_loss.data))

    # if ((loop+1) % record_step) == 0:
    G_loop = ['%03d' % (epoch+1)]*sample_size
    G_sample_ = x_fake[torch.argmax(y_pred), :]
    G_sample = G_sample_.tolist()
    m, s = G_sample_.mean(), G_sample_.std()
    t, p = shapiro(G_sample)  # H0: it is from a normal distribution
    G_title = 'mean: %.2f, std: %.2f, p-value: %.2e' % (m, s, p)
    sns.violinplot(x=G_loop, y=G_sample).set_title(G_title)
    plt.pause(0.001)
    G_loops += G_loop
    G_samples += G_sample
    G_titles += G_title

sns.violinplot(x=G_loops[-10*sample_size:], y=G_samples[-10*sample_size:])
