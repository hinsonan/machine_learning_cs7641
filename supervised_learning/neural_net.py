from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model

def get_model(input_shape):
    inputs = Input(shape=(input_shape))
    layer = Dense(32, activation='relu')(inputs)
    layer = Dense(16, activation='relu')(layer)
    layer = Dense(8, activation='relu')(layer)
    output = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs, output)
    return model
