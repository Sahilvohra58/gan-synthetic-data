import numpy as np
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization


class Generator:
    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        self.W = width
        self.H = height
        self.C = channels
        self.LATENT_SPACE_SIZE = latent_size
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.Generator = self.model()
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

        self.LATENT_SPACE_SIZE = latent_size
        self.latent_space = np.random.normal(0, 1, (self.LATENT_SPACE_SIZE,))

    def model(self, block_starting_size=128, num_blocks=4):
        model = Sequential()
        block_size = block_starting_size
        model.add(Dense(block_size, input_shape=(self.LATENT_SPACE_SIZE, )))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        for i in range(num_blocks-1):
            block_size = block_size * 2
            model.add(Dense(block_size))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.W * self.H * self.C, activation='tanh'))
        model.add(Reshape((self.W, self.H, self.C)))

        return model

    def summary(self):
        return self.Generator.summary()
