#Simple Autoencoder

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import numpy as np
import random

# Number of epochs to train
EPOCHS = 200

# Number of fake datapoints to create                                                                                          
DATA_SIZE = 10000
DATA_NOISE = 0.1
VAL_SIZE = int(0.2 * DATA_SIZE)

# Test Noise level
TEST_NOISE = 0.3

# create autoencoder
aeinput = Input(shape = (1, ), name = 'input')
encoder = Dense(units = 4, activation = 'relu')(aeinput)
encoder = Dense(units = 8, activation = 'relu')(encoder)
encoder = Dense(units = 16, activation = 'relu')(encoder) 
decoder = Dense(units = 8, activation = 'relu')(encoder)
decoder = Dense(units = 4, activation = 'relu')(decoder)
aeoutput = Dense(units = 1, activation = 'tanh')(decoder)
ae = Model(aeinput, aeoutput)
ae.compile(loss = 'mean_squared_error', optimizer = 'sgd')

random.seed(24601)

def noise(scale):
    return (2 * random.uniform(0, 1) - 1) * scale 

def gen_X(data_size, noise_level):
    return [random.uniform(0, 1) + noise(noise_level) for i in range(data_size)]

def gen_WrongX(data_size, noise_level):
    return [random.uniform(0, 1.5) + noise(noise_level) for i in range(data_size)]

X_in = np.array(gen_X(DATA_SIZE, DATA_NOISE))
X_test = np.array(gen_X(VAL_SIZE, DATA_NOISE))

X_noisy = np.array(gen_X(DATA_SIZE, TEST_NOISE))
X_wrong = np.array(gen_WrongX(DATA_SIZE, DATA_NOISE))

ae.fit(x = X_in, y = X_in, batch_size = 100, 
epochs = EPOCHS, validation_data = (X_test, X_test))

clean_loss = ae.evaluate(x = X_in, y = X_in)
test_loss = ae.evaluate(x = X_test, y = X_test)
noisy_loss = ae.evaluate(x = X_noisy, y = X_noisy)
wrong_loss = ae.evaluate(x = X_wrong, y = X_wrong)

print("\n\nClean loss = %3.4f, Test loss = %3.4f Noisy loss = %3.4f, Wrong loss = %3.4f" % 
(clean_loss, test_loss, noisy_loss, wrong_loss))
