from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential


class GAN:
    def __init__(self, discriminator, generator):
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)

        self.Generator = generator

        self.Discriminator = discriminator
        self.Discriminator.trainable = False

        self.gan_model = self.model()
        self.gan_model.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

    def model(self):
        model = Sequential()
        model.add(self.Generator)
        model.add(self.Discriminator)
        return model

    def summary(self):
        return self.gan_model.summary()
