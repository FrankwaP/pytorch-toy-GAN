# -*- coding: utf-8 -*-
"""
Toy model to learn how to play with a GAN model
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

from math import log
import seaborn as sns
import matplotlib.pyplot as plt
# optimizer to play with
from torch.optim import Adam


if torch.cuda.is_available():
    print('Yes! CUDA is available!')
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#%% Functions to generate the data

def generate_real_data(batch_size, sample_size):
    mu, sig = 1., 0.05
    return mu + sig*torch.randn((batch_size, sample_size)).to(device)

def generate_noise(batch_size, noise_size):
    return torch.rand((batch_size, noise_size)).to(device)

#%% Fcuntion for distribution plotting
def results_to_seaborn_plot():
    x_min, x_max = -0.5, 1.25
    def best_sample(x, y):
        return x[torch.argmax(y), :].tolist(), torch.max(y)
    def worst_sample(x, y):
        return x[torch.argmin(y), :].tolist(), torch.min(y)
    
    xy_dic = {'real': (x_real, y_real_pred),
              'fake2': (x_fake2, y_fake_pred2)}

    sample_list = []
    label_list = []
    for ref, xy in xy_dic.items():
        for i, func in enumerate((best_sample, worst_sample)):
            sample, score = func(xy[0], xy[1])
            sample_list += sample
            label = ref+'-'+str(i)+': %.2f' % score
            label_list += [label]*len(sample)
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlim(x_min, x_max)
    sns.violinplot(ax=ax, x=sample_list, y=label_list).set_title('epoch '+str(epoch+1))
    plt.pause(0.001)


#%% Discriminator and Generator

class sort_layer(nn.Module):
    def forward(self, x):
        return x.sort(dim=1)[0]

def get_discriminator(n_input, n_hidden, n_output=1):
    return nn.Sequential(
        sort_layer(),
        nn.Linear(n_input, n_hidden),
        nn.LeakyReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.LeakyReLU(),
        nn.Linear(n_hidden, n_output),
        nn.Sigmoid())

def get_generator(n_input, n_hidden, n_output):
    return nn.Sequential(
        nn.Linear(n_input, n_hidden),
        nn.Tanh(),
        nn.Linear(n_hidden, n_output))


# We use the BCE loss for both Discriminator and Generator
# The original article makes it look more complicated than it is
# by the way the equation are written, but it really is just that:
loss = nn.BCELoss()


#%% Training the GAN
# Don't hesitate to restart a bad training as it is sensitive to the initial random parameters
# You'll hopefully see the fake distributions "crawl" toward the real ones
# then adjust their shapes

batch_size = 10000
sample_size = 1000  # "enough" so the mean value is stable
noise_size = 50  # very impactful parameter!

# The prec_criterion has been added to enforce a minimal precision
# before stoping the training loop of the Discriminator or Generator
# The objective is to test if it helps the game staying balanced
# Of course setting it prec_criterion to a low value gives
# the same behaviour as a "standard" GAN (one training step each)
prec_criterion = 0.5
# Now we can add a global stop criterion based on the number of sub-loop
max_sub_loop = 20
loss_criterion = -log(prec_criterion)
n_epoch = 2000
plot_step = 50

D = get_discriminator(sample_size, 2**3, 1)
D.to(device)
D_optim = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

G = get_generator(noise_size, 2**4, sample_size)
G.to(device)
G_optim = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

y_real = torch.ones((batch_size, 1)).to(device)
y_fake = 1-y_real

# "Bare step" to show initial behaviour
# We notice that the initial behaviour of tthe Generator is quite
# close to a normal distribution with mean 0
epoch = -1
z_fake = generate_noise(batch_size, noise_size)
x_fake2 = G(z_fake)
y_fake_pred2 = D(x_fake2)
x_real = generate_real_data(batch_size, sample_size)
y_real_pred = D(x_real)
results_to_seaborn_plot()

# And here we go…
for epoch in range(n_epoch):
    # Training the Discriminator
    loss_chk = 2*loss_criterion
    i_sub_loop = 1
    while i_sub_loop <= max_sub_loop and loss_chk >= loss_criterion:
        i_sub_loop += 1
        # Discriminator loss on fake data (should be fake)
        z_fake = generate_noise(batch_size, noise_size)
        x_fake1 = G(z_fake)
        y_fake_pred1 = D(x_fake1)
        D_fake_loss = loss(y_fake_pred1, y_fake)
        # Discriminator loss on real data (should be real)
        x_real = generate_real_data(batch_size, sample_size)
        y_real_pred = D(x_real)
        D_real_loss = loss(y_real_pred, y_real)
        ## combining losses
        D_loss = 0.5*(D_real_loss + D_fake_loss)
        loss_chk = D_loss.data
        # fitting Discriminator parameter
        D.zero_grad()
        D_loss.backward()
        D_optim.step()
        ###
        print('Epoch [%d/%d], D_real_loss: %5.3f, D_fake_loss: %5.3f, D_loss: %5.3f, G_loss: %5s'
              % (epoch+1, n_epoch, D_real_loss.data, D_fake_loss.data, D_loss.data, ''))

    if i_sub_loop > max_sub_loop-1:
        print('max_sub_loop exceeded while training the Discriminator')
    # Training the Generator
    loss_chk = 2*loss_criterion
    i_sub_loop = 1
    while i_sub_loop <= max_sub_loop and loss_chk >= loss_criterion:
        i_sub_loop += 1
        # Generator loss on fake data (should be real)
        z_fake = generate_noise(batch_size, noise_size)
        x_fake2 = G(z_fake)
        y_fake_pred2 = D(x_fake2)
        G_loss = loss(y_fake_pred2, y_real)
        loss_chk = G_loss.data
        # fitting Generator parameter
        G.zero_grad()
        G_loss.backward()
        G_optim.step()
        ###
        print('Epoch [%d/%d], D_real_loss: %5s, D_fake_loss: %5s, D_loss: %5s, G_loss: %5.3f'
              % (epoch+1, n_epoch, '', '', '', G_loss.data))
    if i_sub_loop > max_sub_loop-1:
        print('max_sub_loop exceeded while training the Generator')
    # End of the loop
    if ((epoch+1) % plot_step) == 0:
        results_to_seaborn_plot()
