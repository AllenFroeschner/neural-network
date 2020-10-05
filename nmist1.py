
# found base prog here: https://github.com/tabshaikh/Deep-Learning-with-tf.keras-with-mnist
# added fashion from here: https://www.tensorflow.org/tutorials/keras/basic_classification
# many examples here: https://www.tensorflow.org/tutorials
# if you have ever used photomath this is the code for it


processNumberSet = True # process fashion dataset if false


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



#tf.__version__ - used to check the version of your tensorflow

if processNumberSet:
	mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits from 0-9
	(x_train,y_train), (x_test,y_test) = mnist.load_data() #divide training and testing data
else:
	fashion_mnist = tf.keras.datasets.fashion_mnist # 28x28 images of clothing
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # divide train and test


	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print shape of data
print("x_train shape", np.array(x_train).shape, "y_train shape", np.array(y_train).shape)
print("x_test shape", np.array(x_test).shape, "y_test shape", np.array(y_test).shape)
print()

x_train = tf.keras.utils.normalize(x_train,axis=1) # normalizing 
x_test = tf.keras.utils.normalize(x_test,axis=1)


# setup the neural network 
model = tf.keras.models.Sequential() # using the basic feed forward ie Sequential model
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(175, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(175, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics =['accuracy']  )


print('Training...')
model.fit(x_train,y_train,epochs = 3) # train the neural network

val_loss, val_accuracy = model.evaluate(x_test,y_test)
print("done - loss and accuracy:",val_loss, val_accuracy) #printing test cost and accuracy
print()


# predicte first few items in test set and display image and prediction
predictions = model.predict([x_test])

for i in range(5):
	print("prediction:",np.argmax(predictions[i]) ) #showing the prediction of 1st test example
	if not processNumberSet:
		print(class_names[np.argmax(predictions[i]) ])


	plt.imshow(x_test[i], cmap=plt.cm.binary) #showing the test example
	plt.show()

