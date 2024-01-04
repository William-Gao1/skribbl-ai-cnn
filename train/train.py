import os
import tqdm

scratch = os.path.join(os.path.expanduser('~'), "scratch")

import numpy as np
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


target_shape = (64, 64)
num_per_category = 1000

loc = os.path.join(scratch, f"{target_shape[0]}-pre-quickdraw/")
files = sorted([os.path.join(loc, c) for c in os.listdir(loc)])

save_model_name = f"../quickdraw-{target_shape[0]}.model"

X = np.empty((len(files) * num_per_category, target_shape[0], target_shape[1]))

y = np.array([])

for idx, file in tqdm.tqdm(list(enumerate(files))):
  try:
    f = np.load(file)
    X[num_per_category*idx:num_per_category*(idx+1)] = f
    y = np.append(y, np.ones(len(f)) * idx)
  except Exception as e:
    print(file)
    print(e)
    exit(1)
  
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# one hot encode outputs
y_train_cnn = to_categorical(y_train)
y_test_cnn = to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], target_shape[0], target_shape[1], 1).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], target_shape[0], target_shape[1], 1).astype('float32')
s = X_train_cnn.shape
print (s, num_classes)

cnn_model = Sequential([
    BatchNormalization(),

    Conv2D(24, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(48, kernel_size=(3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    
    BatchNormalization(),
    
    Flatten(),

    Dense(700, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(500, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(400, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# build the model
model = cnn_model
# Fit the model
history = model.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=30, batch_size=50)
# Final evaluation of the model
scores = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Final CNN accuracy: ', scores[1]*100, "%")

# Save weights
model.save(save_model_name)
print("Model is saved")