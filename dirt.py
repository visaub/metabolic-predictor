# def rec(j,l):  #Recursive function
# 	if j==1:
# 		return l
# 	lf=[]
# 	l0=['Weight','Load','Velocity','Slope']
# 	for e in l:
# 		index=-1
# 		for i0 in range(len(l0)):
# 			if e[0]==l0[i0]:
# 				index=i0
# 				for i in range(index+1):
# 					lf+=rec(j-1,[[l0[i]]+e])
# 	return lf

# print(rec(2,[['Weight'],['Load'],['Velocity'],['Slope']]))


# import numpy as np
# from keras.utils import to_categorical
# from keras import models
# from keras import layers
# from keras.datasets import imdb
# (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
# data = np.concatenate((training_data, testing_data), axis=0)
# targets = np.concatenate((training_targets, testing_targets), axis=0)
# def vectorize(sequences, dimension = 10000):
#  results = np.zeros((len(sequences), dimension))
#  for i, sequence in enumerate(sequences):
#   results[i, sequence] = 1
#  return results
 
# data = vectorize(data)
# targets = np.array(targets).astype("float32")
# test_x = data[:10000]
# test_y = targets[:10000]
# train_x = data[10000:]
# train_y = targets[10000:]
# model = models.Sequential()
# # Input - Layer
# model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
# # Hidden - Layers
# model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
# model.add(layers.Dense(50, activation = "relu"))
# model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
# model.add(layers.Dense(50, activation = "relu"))
# # Output- Layer
# model.add(layers.Dense(1, activation = "sigmoid"))
# model.summary()
# # compiling the model
# model.compile(
#  optimizer = "adam",
#  loss = "binary_crossentropy",
#  metrics = ["accuracy"]
# )
# results = model.fit(
#  train_x, train_y,
#  epochs= 2,
#  batch_size = 500,
#  validation_data = (test_x, test_y)
# )
# print("Test-Accuracy:", np.mean(results.history["val_acc"]))

# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout

# # Generate dummy data
# x_train = np.random.random((1000, 20))
# y_train = np.random.randint(2, size=(1000, 1))
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(2, size=(100, 1))

# model = Sequential()
# model.add(Dense(64, input_dim=20, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           epochs=20,
#           batch_size=128)
# score = model.evaluate(x_test, y_test, batch_size=128)

