from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from dummypy.GAN import GAN

df = pd.read_csv("data/adult.csv")
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
gan.train(epochs=50)

###########################

generated_df = gan.Generator.generate(instances=100)
generated_df = pd.DataFrame(generated_df)
generated_df.describe()
df.describe()

############################


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam

BLOCK_SIZE = 100
LATENT_SPACE_SIZE=10
ALPHA = 0.2
MOMENTUM = 0.8
BLOCKS = 3
OUTPUT_SHAPE = (6,)
model = Sequential()
model.add(Dense(BLOCK_SIZE, input_shape=(LATENT_SPACE_SIZE,)))
model.add(LeakyReLU(alpha=ALPHA))
model.add(BatchNormalization(momentum=MOMENTUM))

for i in range(BLOCKS-1):
    BLOCK_SIZE = int(BLOCK_SIZE / 2)
    model.add(Dense(BLOCK_SIZE))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(BatchNormalization(momentum=MOMENTUM))

model.add(Dense(OUTPUT_SHAPE[0], activation='tanh'))
model.add(Reshape((OUTPUT_SHAPE[0], )))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, decay=8e-9))

gan.Generator.GeneratorModel = model


###################################

from dummypy.discriminator import Discriminator
model = Discriminator(input_shape=(6,)).DiscriminatorModel
gan.Discriminator.DiscriminatorModel = model


#############################

