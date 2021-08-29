from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def get_model(input_shape):
    inputs = Input(shape=(input_shape))
    layer = Dense(32, activation='relu')(inputs)
    layer = Dense(16, activation='relu')(layer)
    layer = Dense(4, activation='relu')(layer)
    output = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs, output)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,loss='binary_crossentropy')
    return model
