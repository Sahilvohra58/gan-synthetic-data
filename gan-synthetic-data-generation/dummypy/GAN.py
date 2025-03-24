from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
import pandas as pd
import tensorflow as tf
import numpy as np

from dummypy.generator import Generator
from dummypy.discriminator import Discriminator


class GAN:
    def __init__(self,
                 dataset: pd.DataFrame = None,
                 discriminator_model=None,
                 generator_model=None,
                 optimizer=None,
                 loss=None,
                 print_summary=False):

        self.OPTIMIZER = Adam(learning_rate=0.0002, decay=8e-9) if optimizer is None else optimizer
        self.LOSS = 'binary_crossentropy' if loss is None else loss

        self.DATASET = dataset
        self.FEATURES = (self.DATASET.shape[1],)
        self.TF_DATASET = tf.constant(self.DATASET.values)

        if generator_model is None:
            self.Generator = Generator(output_shape=self.FEATURES)
            self.GeneratorModel = self.Generator.GeneratorModel
        else:
            self.GeneratorModel = generator_model
            self.Generator = Generator(output_shape=self.FEATURES)
            self.Generator.GeneratorModel = self.GeneratorModel

        if discriminator_model is None:
            self.Discriminator = Discriminator(input_shape=self.FEATURES)
            self.DiscriminatorModel = self.Discriminator.DiscriminatorModel
        else:
            self.DiscriminatorModel = discriminator_model
            self.Discriminator = Discriminator(input_shape=self.FEATURES)
            self.Discriminator.DiscriminatorModel = self.DiscriminatorModel

        self.Discriminator.trainable = False

        self.gan_model = self.model()
        self.gan_model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER)

        if print_summary:
            print("------------------ GAN Summary ------------------")
            print(self.summary())
            print("------------------ Generator Summary ------------------")
            print(self.Generator.summary())
            print("------------------ Discriminator Summary ------------------")
            print(self.Discriminator.summary())

    def model(self):
        model = Sequential()
        model.add(self.GeneratorModel)
        model.add(self.DiscriminatorModel)
        return model

    def summary(self):
        return self.gan_model.summary()

    def train(self, epochs):
        for e in range(epochs):
            real_labels = tf.ones(shape=(len(self.TF_DATASET), 1), dtype=tf.dtypes.int32)

            # Grab generated data
            latent_space_samples = self.Generator.sample_latent_space(len(self.TF_DATASET))
            generated_features = self.GeneratorModel.predict(latent_space_samples)
            generated_labels = tf.zeros(shape=(len(generated_features), 1), dtype=tf.dtypes.int32)

            # Combine the data for discriminator
            combined_features = tf.concat([self.TF_DATASET, generated_features], 0)
            combined_labels = tf.concat([real_labels, generated_labels], 0)

            discriminator_loss = self.DiscriminatorModel.train_on_batch(combined_features, combined_labels)[0]

            # Generate Noise
            latent_space_samples = self.Generator.sample_latent_space(len(generated_features))
            generated_labels = np.ones([len(generated_features), 1])
            generator_loss = self.gan_model.train_on_batch(latent_space_samples, generated_labels)

            print('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss: ' + str(
                discriminator_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')


if __name__ == "__main__":
    # import numpy as np
    # sample_df = np.random.normal(0, 1, (15, 5))
    # gan = GAN(dataset=sample_df, print_summary=True)
    # print(gan.summary())

    # Preprocessing
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    df = pd.read_csv("../data/adult.csv")
    le = preprocessing.LabelEncoder()
    drop_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex',
                    'native.country', 'income']
    df = df.drop(drop_columns, axis=1)
    columns = df.columns
    # for i in drop_columns:
    #     df[i] = le.fit_transform(df[i].astype(str))
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=columns, dtype=np.float32)

    # Training
    gan = GAN(dataset=df)
    gan.train(epochs=150)
