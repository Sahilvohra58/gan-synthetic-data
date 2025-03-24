from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam


class Discriminator:
    def __init__(self, width=28, height=28, channels=1, latent_size=100):
        self.CAPACITY = width * height * channels
        self.SHAPE = (width, height, channels)
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.Discriminator = self.model()
        self.Discriminator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])
        self.summary()

    def model(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.SHAPE))
        model.add(Dense(self.CAPACITY, input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(int(self.CAPACITY/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def summary(self):
        return self.Discriminator.summary()
