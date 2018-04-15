# 準備資料
# 1. import required module

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
X = np.linspace(-1, 1, 500)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (500, ))
X_train, Y_train = X[:250], Y[:250]
X_test, Y_test = X[250:], Y[250:]

#print (X, Y)

model = Sequential()
dense = Dense(units=1, input_dim=1)
model.add(dense)
model.compile(loss='mse', optimizer='sgd')

model.fit(x=X_train, y=Y_train, epochs=300)

for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: {}'.format(cost))


cost = model.evaluate(X_test, Y_test, batch_size=40)
print()
print('accuracy = ', cost)

W, b = model.layers[0].get_weights()
print("weights = {}, biases= {}".format(W, b))