from dummypy.GAN import GAN
from dummypy.generator import Generator
from dummypy.discriminator import Discriminator

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame
import tensorflow as tf


class Trainer:
    def __init__(self, dataset: DataFrame, epochs=50, latent_size=10):
        self.LATENT_SPACE_SIZE = latent_size
        self.EPOCHS = epochs
        self.DATASET = dataset
        self.TF_DATASET = tf.constant(self.DATASET.values)
        self.FEATURES = self.DATASET.shape[1]

        self.generator = Generator(output_shape=(self.FEATURES,), latent_size=self.LATENT_SPACE_SIZE)
        self.discriminator = Discriminator(input_shape=(self.FEATURES,))
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)

    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances, self.LATENT_SPACE_SIZE))

    def train(self):
        for e in range(self.EPOCHS):
            real_labels = tf.ones(shape=(len(self.TF_DATASET), 1), dtype=tf.dtypes.int32)

            # Grab generated data
            latent_space_samples = self.sample_latent_space(len(self.TF_DATASET))
            generated_features = self.generator.Generator.predict(latent_space_samples)
            generated_labels = tf.zeros(shape=(len(generated_features), 1), dtype=tf.dtypes.int32)

            # Combine the data for discriminator
            combined_features = tf.concat([self.TF_DATASET, generated_features], 0)
            combined_labels = tf.concat([real_labels, generated_labels], 0)

            discriminator_loss = self.discriminator.Discriminator.train_on_batch(combined_features, combined_labels)[0]

            # Generate Noise
            latent_space_samples = self.sample_latent_space(len(generated_features))
            generated_labels = np.ones([len(generated_features), 1])
            generator_loss = self.gan.gan_model.train_on_batch(latent_space_samples, generated_labels)

            print('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss: ' + str(
                discriminator_loss) + '], [ Generator :: Loss: ' + str(generator_loss) + ']')


if __name__ == "__main__":

    # Preprocessing
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
    trainer = Trainer(dataset=df, epochs=150, latent_size=10)
    trainer.train()
