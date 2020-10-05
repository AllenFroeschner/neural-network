from keras.models import Sequential
from keras.layers import Dense

from math import sin, cos
import sys

import numpy as np

np.random.seed(51337)  # for reproducibility of random number generator
np.set_printoptions(precision=8)   # number of decimals in numpy formatting
np.set_printoptions(suppress=True) # no scientific notation printing numpy arrays


# setup key constants
functionToApprox = 11
#functionToApprox = int(sys.argv[1]) # take this from command line

numBatches  = 10000     # total number of batches to run
rowsToTrain = 500       # rows to train per batch - this is also the 'mini-batch' size
rowsToTest  = 1000      # rows to test
numEpochs   = 1         # how many times to train on each set of data

inputsPerRow  = 50      # inputs should be [at least] a few more than outputs
outputsPerRow = 1      

nnLayer1=0              # number of neurons per hidden layer.  0 = deactivate layer
nnLayer2=0
nnLayer3=0
nnLayer4=0

activationType='relu'  # relu, selu, sigmoid, etc = neuron activation
learningRate = .0001   # generally between .001 and .0001 are good starting choices

numScale = 5           # input number range: -numScale to numScale

col1 = 0               # map program's symbolic columns to actual columns
col2 = 1
col3 = 2

displayWeights = True # display weights and bias' after training


#calc remainder constants
rowsPerBatch = rowsToTrain + rowsToTest
assert inputsPerRow - outputsPerRow > 1   # declare this to be true (errors is not)



#build the untrained neural network
model = Sequential()

if nnLayer1:
	model.add(Dense(nnLayer1, activation=activationType))
if nnLayer2:
	model.add(Dense(nnLayer2, activation=activationType))
if nnLayer3:
	model.add(Dense(nnLayer3, activation=activationType))
if nnLayer4:
	model.add(Dense(nnLayer3, activation=activationType))

model.add(Dense(outputsPerRow, activation='linear'))
model.compile(loss='mse', optimizer='adam', lr=learningRate)




# init Y array
Y = np.zeros((rowsPerBatch, outputsPerRow) )

# loop to create data, train and report
updateEvery = 100

for batchCnt in range(0, numBatches):

	# *** X is the input to the neural network - setup matrix randomly between 0 and 1
	X = np.random.rand(rowsPerBatch, inputsPerRow)

	# two different experiment types. Experiment 1 is always identity function
	experiment = 2  # experiment 1 = fast identity, experiment 2 = other functions
	if experiment == 1:
		X = X * 100 - 50
		X = np.log(np.abs(X) )
		Y = X[:,:outputsPerRow]

	else:
		X = X * (numScale * 2) - numScale      #scale X

		# the first time through, build the test data
		if batchCnt == 0:
			rowsToBuild = rowsToTrain + rowsToTest
			print('generating testing data')
		else:
			rowsToBuild = rowsToTrain

		for i in range(0, rowsToBuild):

			# map columns to groups of columns
			col1Group = X[i,col1:col1+outputsPerRow]
			col2Group = X[i,col2:col2+outputsPerRow]
			col3Group = X[i,col3:col3+outputsPerRow]


			# *** create Y the groundtruth output to train on input 
			if functionToApprox == 1:
				Y[i] = col1Group  # identity
				desc = 'identity of 1st col'

			elif functionToApprox == 2:
				Y[i] = np.abs(col1Group) # absolute value
				desc = 'absolute value of 1st col'

			elif functionToApprox == 3:
				Y[i] = col1Group + col2Group
				desc = '1st col + 2nd col'

			elif functionToApprox == 4:
				Y[i] = col1Group + (col2Group * col3Group)
				desc = '1st col + (2nd col * 3rd col)'

			elif functionToApprox == 5:
				Y[i] = np.sin( col1Group * 3.14 / 2 ) * col2Group
				desc = 'sin(1st col * pi/2) * 2nd col'

			elif functionToApprox == 6:
				Y[i] = col1Group / col2Group 
				desc = 'division 1st col / 2nd col - this function does not work'

			elif functionToApprox == 7:
				Y[i] = np.cos( col1Group  *3.14 /2 ) * col2Group
				desc = 'cos(1st col * pi/2) * 2nd col'

			elif functionToApprox == 8:
				Y[i] = 3.14 * np.square( col1Group )
				desc = 'area of circle, 1st col = radius'

			elif functionToApprox == 9:
				Y[i] = np.sqrt( np.square(col1Group) + np.square(col2Group) )
				desc = 'pythag theorm 1st col=a, 2nd col=b'

			elif functionToApprox == 10:
				Y[i] = np.log( np.abs(col1Group) )
				desc = 'log base e of abs(1st col)'

			elif functionToApprox == 11:
				Y[i] = col1Group * 3.1415926
				desc = 'circ of circle 1st col=diameter'

			elif functionToApprox == 12:
				Y[i] = col1Group * col1Group * col1Group * 4 / 3 * 3.14
				desc = 'volume of a sphere 1st col=radius'

			elif functionToApprox == 13:
				Y[i] = col1Group * 5 # *5 or whatever
				desc = 'my custom function'

	# first time through, split out the test data
	if batchCnt == 0:
		X_test, Y_test = X[rowsToTrain:,:], Y[rowsToTrain:,:]

	# split one batch of train/test data
	X_train, Y_train = X[:rowsToTrain,:], Y[:rowsToTrain,:]

	# train the network on one batch
	for step in range(numEpochs):
		cost = model.train_on_batch(X_train, Y_train)   # <- train the neural network

	# report occasionally
	if batchCnt % updateEvery == 0:
		print('batch: ', batchCnt,'- total TRAIN rows :', rowsToTrain* batchCnt)
		print()
		print( 'function to approximate: ', desc)
		print()

		# split data into test X,Y groups
		#X_test, Y_test = X[rowsToTrain:,:], Y[rowsToTrain:,:]

		sampleRow = int(batchCnt / 2000) +1 # switch example rows every 2000 batches

		print('----- SAMPLE ROW NUMBER', sampleRow, 'FROM TEST SET -----')

		print('INPUT data - first 5 columns of', inputsPerRow )
		print('                   ', X_test[sampleRow:sampleRow+1,:5])
		print()

		#print("below=output groudtruth - first", min(5,outputsPerRow), "columns of",outputsPerRow)
		print("OUTPUT data - first", min(5, outputsPerRow), "columns of", outputsPerRow)
		print('       groundtruth:', Y_test[sampleRow:sampleRow+1,:5])

		Y_pred = model.predict(X_test)
		print('        prediction:',Y_pred[sampleRow:sampleRow+1,:5])
		print()

		diff = np.absolute(Y_test - Y_pred)
		avgAbsDiff = np.sum( diff) / ( Y_test.shape[0] * Y_test.shape[1]) 

		print('average difference of all', rowsToTest, 'TEST rows x', outputsPerRow, 'outputs per row:' ,avgAbsDiff) 
		print('___________________________________________'*2)
		print()




# here is where you would save the trained model


# display model layers
print(model.summary())

#dump weights and biases of network
if displayWeights:
	print('layer 1')
	W, b = model.layers[0].get_weights()
	print('Weights=', W, '\nbiases=', b)

	if nnLayer1:
		print('layer 2')
		W, b = model.layers[1].get_weights()
		print('Weights=', W, '\nbiases=', b)

