# 進行資料預處理
# 1. import required module
import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)

# 2. read minst data
from keras.datasets import mnist
(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()

print ('train image =', len(X_train_image), X_train_image.shape)
print (' test image =', len(X_test_image))

import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

#plot_image(X_train_image[0])
print (y_train_label[0])

# 3. 將 feature convert to 784 float number
x_Train = X_train_image.reshape(60000, 784).astype('float32')
x_Test  = X_test_image.reshape(10000, 784).astype('float32')
print ('x_Train = ' , x_Train.shape)
print ('x_Train_image 0 = ', X_train_image[0])

# 4. normalize feature value
x_Train_normalize = x_Train / 255
x_Test_normalize  = x_Test / 255
print (x_Test_normalize[0])

# 5. 將label 以 one-hot encoding
print (y_train_label[:5])
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot  = np_utils.to_categorical(y_test_label)
print (y_TrainOneHot[:5])

###########################################################################

# 建立模型
# 1. import required module
from keras.models import Sequential
from keras.layers import Dense

# 2. 建立 sequential 模型
model = Sequential()

# 3. 建立輸入層
model.add(Dense(units=256,
               input_dim=784,
               kernel_initializer='normal',
               activation='relu'
        ))

# 4. 建立輸出層
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'
               ))

# 5. 查看summary
print (model.summary())

# 進行訓練
# 1. 定義訓練方式
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# 2. 開始訓練
train_history = model.fit(x=x_Train_normalize,
                          y=y_TrainOneHot, validation_split=0.2,
                          epochs=10, batch_size=200, verbose=2)

                          
# 以測試資料評估模型準確率
# 1. 評估模型準確率
scores = model.evaluate(x_Test_normalize, y_TestOneHot)
print()
print('accuracy = ', scores[1])

# 進行預測
# 1. 執行預測
prediction = model.predict_classes(x_Test)
print(prediction[:10])