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
# optimizer to play with
from torch.optim import Adam


if torch.cuda.is_available():
    print('Yes! CUDA is available!')
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#%% Functions to generate the data

def generate_real_data(batch_size, sample_size):
    mu, sig = 2., 0.5
    return mu + sig*torch.randn((batch_size, sample_size)).to(device)

def generate_noise(batch_size, noise_size):
    return torch.rand((batch_size, noise_size)).to(device)

#%% Fcuntion for distribution plotting
def results_to_seaborn_plot():
    def best_sample(x, y):
        return x[torch.argmax(y), :].tolist(), torch.max(y)
    def worst_sample(x, y):
        return x[torch.argmin(y), :].tolist(), torch.min(y)
    
    xy_dic = {'real': (x_real, y_real_pred), 'fake1': (x_fake_comp, y_fake_pred_comp),
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
    sns.violinplot(ax=ax, x=sample_list, y=label_list).set_title('epoch '+str(epoch+1))
    plt.pause(0.001)


#%% Discriminator and Generator

class sort_layer(nn.Module):
    def forward(self, x):
        return x.sort(dim=1)[0]

def get_discriminator(n_input, n_hidden, n_output=1):
    return nn.Sequential(
        sort_layer(),
        # nn.Dropout(0.9),
        nn.Linear(n_input, n_output), 
        nn.Sigmoid())

def get_generator(n_input, n_hidden, n_output):
    return nn.Sequential(
        nn.Linear(n_input, n_hidden),
        nn.Tanh(),
        nn.Linear(n_hidden, n_output))


# We use the BCE loss for both Discriminator and Generator
# This doesn't correspond to equation 1 and algorithm 1, regarding the Generator
# (watch out: the equations use logs and not -logs like the BCE)
# However: "Rather than training G to minimize log(1 - D(G(z))) we can train G to maximize logD(G(z)).
# This objective function results in the same fixed point of the dynamics of G and D
# but provides much stronger gradients early in learning."
# Using the BCE loss for the generator does this.
loss = nn.BCELoss()


#%% Training the GAN: batch level

batch_size = 10000
noise_size = 100
sample_size = 1000
n_epoch = 100
plot_step = 1

D = get_discriminator(sample_size, 2**3, 1)
D.to(device)

G = get_generator(noise_size, 2**4, sample_size)
G.to(device)

y_real = torch.ones((batch_size, 1)).to(device)
y_fake = 1-y_real

loss_limit = 0.11  # correspond to a 90% success mean: -ln(0.9) == 0.11

x_fake_comp = None
y_fake_pred_comp = None

for epoch in range(n_epoch):
    # Training the Discriminator
    D_optim = Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_chk = 2*loss_limit
    while loss_chk > loss_limit:
        # Discriminator loss on fake data (should be fake)
        z_fake = generate_noise(batch_size, noise_size)
        x_fake1 = G(z_fake)
        # if old_x_fake1 is None:
        y_fake_pred1 = D(x_fake1)
        D_fake_loss = loss(y_fake_pred1, y_fake)
        # else:            
        #     y_fake_pred1 = D(torch.cat((x_fake1, old_x_fake1)))
        #     D_fake_loss = loss(y_fake_pred1, torch.cat((y_fake, y_fake)))
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
        print('Epoch [%d/%d], D_real_loss: %5.2f, D_fake_loss: %5.2f, D_loss: %5.2f, G_loss: %5s'
              % (epoch+1, n_epoch, D_real_loss.data, D_fake_loss.data, D_loss.data, ''))
    old_x_fake1 = x_fake1.detach()
    # Training the Generator
    loss_chk = 2*loss_limit
    G_optim = Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    while loss_chk > loss_limit:
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
        print('Epoch [%d/%d], D_real_loss: %5s, D_fake_loss: %5s, D_loss: %5s, G_loss: %5.2f'
              % (epoch+1, n_epoch, '', '', '', G_loss.data))
    if ((epoch+1) % plot_step) == 0:
        if x_fake_comp is None:
            x_fake_comp = x_fake1
            y_fake_pred_comp = y_fake_pred1
        results_to_seaborn_plot()
        x_fake_comp = x_fake2
        y_fake_pred_comp = y_fake_pred2


#%%
tmp1 = y_fake_pred2.tolist()  # predicted values in list (for Excel)
tmploss = nn.BCELoss(reduction='none')  # redefining a loss giving the whole BCEloss tensor
tmp2 = tmploss(y_fake_pred2, y_real).tolist()  # BCEloss values in list (for Exel)