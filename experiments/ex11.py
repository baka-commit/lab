# -*- coding: utf-8 -*-
# Experiment 11 - RNN model to perform sentiment analysis
# ========================================================

# Sentiment analysis of movie reviews using RNN
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing import sequence

NUM_WORDS = 5000  # Constant variable: 5000 popular words in the dataset

# Load IMDB dataset from keras datasets
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
print(x_train.shape)
print(x_test.shape)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

print('Review \n', x_train[0])  # 0th review details
print('Label \n', y_train[0])   # 1-means positive review and 0-means negative review

# Converting the words into integers
word_to_id = imdb.get_word_index()
word_to_id

# id to word method. Its a reverse of previous operation
id_to_word = {value: key for key, value in word_to_id.items()}

# x_train[0] preview
for id in x_train[0]:
    word = id_to_word.get(id, '?')
    print(word, end=' ')  # Give Space

print('\nlen(x_train[0]):', len(x_train[0]))

# 218 is heavy length. So, apply truncate and pad sequences to maximum of 200 only
max_words = 200
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)

print('After padding - x_train len:', len(x_train[0]))
print('After padding - x_test len:', len(x_test[0]))

#Build the model
model = Sequential(
 [
 Embedding(NUM_WORDS,64,input_length=max_words),  LSTM(200),
 Dense(1,activation='sigmoid')
 ]
)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display the model summary
model.summary()

# Training the model
model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=5,
    verbose=1,
    validation_data=(x_test, y_test)
)

# Evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Let us use the model and predict whether a given review is positive or negative
tmp = sequence.pad_sequences([x_train[0]], maxlen=max_words)
prediction = model.predict(tmp)
print(prediction)

# If the output is below 50% then negative review and if the output is more than 50% then positive review
