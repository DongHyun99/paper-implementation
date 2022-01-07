from typing import Generator
from google.protobuf import descriptor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Input, Flatten
from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np

# defimne input latnet dimention & image shape
latent_dim = 10
img_shape = (28, 28, 3) # rows, cols, channels

# define Generator model 
def generator_model():

    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    
    return model

def discriminator_model():

    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model

generator = generator_model()
discriminator = discriminator_model()
adam = Adam(lr=9.0002, beta_1=0.5)

discriminator.compile(loss = 'binary_crossentropy', optimizer=adam)

discriminator.trainalbe = False
gan_input = Input(shape=latent_dim)
x = generator(input=gan_input)
output = discriminator(x)

gan = Model(gan_input, output)

gan.summary()