from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LeakyReLU
from math import prod


class Discriminator:
    def __init__(self,
                 input_shape,
                 optimizer=None,
                 blocks=1,
                 block_size=None,
                 loss=None,
                 metrics=None,
                 slope=0.2,
                 print_summary=False):

        self.OPTIMIZER = Adam(learning_rate=0.0002, decay=8e-9) if optimizer is None else optimizer
        self.BLOCKS = blocks
        self.ALPHA = slope
        self.LOSS = 'binary_crossentropy' if loss is None else loss
        self.METRICS = ['accuracy'] if metrics is None else metrics

        self.INPUT_SHAPE = input_shape

        if not block_size:
            self.BLOCK_SIZE = input_shape[0] if len(input_shape) == 1 else prod(input_shape)
        else:
            self.BLOCK_SIZE = block_size

        self.DiscriminatorModel = self.model()
        self.DiscriminatorModel.compile(loss=self.LOSS, optimizer=self.OPTIMIZER, metrics=self.METRICS)
        if print_summary:
            print(self.summary())

    def model(self):
        model = Sequential()

        for _ in range(self.BLOCKS):
            model.add(Flatten(input_shape=self.INPUT_SHAPE))
            model.add(Dense(self.BLOCK_SIZE))
            model.add(LeakyReLU(alpha=self.ALPHA))
            self.BLOCK_SIZE = int(self.BLOCK_SIZE / 2)

        model.add(Dense(int(self.BLOCK_SIZE)))
        model.add(LeakyReLU(alpha=self.ALPHA))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def summary(self):
        return self.DiscriminatorModel.summary()


if __name__ == "__main__":
    discriminator = Discriminator(
        input_shape=(10, 1)
    )
    print(discriminator.summary())
