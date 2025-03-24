import numpy as np
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from math import prod


class Generator:
    def __init__(self,
                 output_shape,
                 latent_size=None,
                 optimizer=None,
                 loss=None,
                 blocks=1,
                 block_size=None,
                 momentum=0.8,
                 slope=0.2,
                 print_summary=False
                 ):

        self.BLOCKS = blocks
        self.MOMENTUM = momentum
        self.ALPHA = slope

        self.OPTIMIZER = Adam(learning_rate=0.0002, decay=8e-9) if optimizer is None else optimizer
        self.OUTPUT_SHAPE = output_shape
        self.LOSS = 'binary_crossentropy' if loss is None else loss
        self.BLOCK_SIZE = int(prod(output_shape)*(blocks+1)) if block_size is None else block_size
        self.LATENT_SPACE_SIZE = int(1.5*self.BLOCK_SIZE) if latent_size is None else latent_size

        self.GeneratorModel = self.model()
        self.GeneratorModel.compile(loss=self.LOSS, optimizer=self.OPTIMIZER)

        self.LATENT_SAMPLE = np.random.normal(0, 1, (self.LATENT_SPACE_SIZE,))

        if print_summary:
            print(Generator.summary())

    def model(self):
        model = Sequential()

        model.add(Dense(self.BLOCK_SIZE, input_shape=(self.LATENT_SPACE_SIZE,)))
        model.add(LeakyReLU(alpha=self.ALPHA))
        model.add(BatchNormalization(momentum=self.MOMENTUM))

        for i in range(self.BLOCKS-1):
            self.BLOCK_SIZE = int(self.BLOCK_SIZE / 2)
            model.add(Dense(self.BLOCK_SIZE))
            model.add(LeakyReLU(alpha=self.ALPHA))
            model.add(BatchNormalization(momentum=self.MOMENTUM))

        model.add(Dense(self.OUTPUT_SHAPE[0], activation='tanh'))
        model.add(Reshape((self.OUTPUT_SHAPE[0], )))

        return model

    def summary(self):
        return self.GeneratorModel.summary()

    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, self.LATENT_SPACE_SIZE))

    def generate(self, instances):
        latent_samples = self.sample_latent_space(instances=instances)
        return self.GeneratorModel.predict(latent_samples)


if __name__ == "__main__":
    generator = Generator(
        output_shape=(10, 1)
    )
    print(generator.summary())
