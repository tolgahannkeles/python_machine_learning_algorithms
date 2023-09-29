import pandas as pd
import numpy as np
from scipy import stats
import keras

from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import SimpleRNN, Dense, Activation


(x_train, y_train), (x_test, y_test)= imdb.load_data(path="ibdb.npz", num_words=None, skip_top=0, maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3)
num_word=15000
max_len=130
x_train=pad_sequences(x_train)
x_test=pad_sequences(x_test)

rnn= Sequential()

rnn.add(Embedding(num_word, 32, input_length=len(x_train[0])))
rnn.add(SimpleRNN(16, input_shape=(num_word,max_len), return_sequences=False, activation="relu" ))
rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))
print(rnn.summary())


rnn.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

rnn.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=5, batch_size=128, verbose=1)

accuracy= rnn.evaluate(x_test, y_test)
print(accuracy)