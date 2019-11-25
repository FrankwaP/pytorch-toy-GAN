# pytorch-toy-GAN

Toy model to learn how to mess with a GAN model (see: https://arxiv.org/abs/1406.2661)
The training data are generated using a normal distribution, meaning:  
    - The generator is trained to generate fake normal distributions from noise (uniform distribution).  
    - The discriminator is trained to detect if normal distributions are real or fake.  

Both distributions -- normal and noise/uniform -- are generated on the fly.
It's nothing special regarding the noise distribution.
But regarding the normal distributions, it has 2 advantages:  
    - this simulate infinite data  
    - this makes the code much shorter  :-)
