# 進行資料預處理
# 1. import required module
import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)

# 2. read minst data
from keras.datasets import mnist
(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()

# 3. 將 features convert to 4D matrix
x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28, 1).astype('float32')
x_Test4D  = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')

# 4. normalize features
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

# 5. onehot encoding
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot  = np_utils.to_categorical(y_Test)

###########################################################################

# 建立模型
# 1. import required module
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

# 2. create model
model = Sequential()

# 3. create convolution layer & pool layer
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))

# create pool layer 1
model.add(MaxPool2D(pool_size=(2,2)))

# 4. create convolution layer 2
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))

# create pool layer 2
model.add(MaxPool2D(pool_size=(2,2)))

# add dropout
model.add(Dropout(0.25))

# 5. create flattern layer
model.add(Flatten())

# create hidden layer
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

print(model.summary())

# 進行訓練
# 1. 定義訓練方式
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# 2. 開始訓練
train_history = model.fit(x=x_Train4D_normalize,
                          y=y_TrainOneHot, validation_split=0.2,
                          epochs=10, batch_size=200, verbose=2)

                          
# 以測試資料評估模型準確率
# 1. 評估模型準確率
scores = model.evaluate(x_Test4D_normalize, y_TestOneHot)
print()
print('accuracy = ', scores[1])

# 進行預測
# 1. 執行預測
prediction = model.predict_classes(x_Test4D_normalize)
print(prediction[:10])