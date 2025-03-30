"""
https://www.adamdavid.dev/

Multi-Class Classification given a time-series of a patient's heart's electrical activity recorded using an electrocardiogram.

Two Classes:                   One-Hot Encoding
  Myocardial Infraction:  -1        [1, 0] 
  Normal Heartbeat:        1        [0, 1]

Using a Convolutional Neural Network and a Recurrent Neural Network.

Dataset Details:
This data set was formatted by R. Olszewski as part of his PhD thesis (see [1]). Each series traces the electrical activity recorded during one heartbeat. The two classes are a normal heartbeat versus a myocardial infarction event (heart attack due to prolonged cardiac ischemia).

Train size: 100

Test size: 100

Missing value: No

Number of classses: 2

Time series length: 96

Data donated by Robert Olszewski (see [1], [2]).

[1] Olszewski, Robert T. Generalized feature extraction for structural pattern recognition in time-series data. No. CMU-CS-01-108. Carnegie Mellow University, School of Computer Science, 2001. URL https://www.cs.cmu.edu/~bobski/pubs/tr01108-twosided.pdf

[2] http://www.timeseriesclassification.com/description.php?Dataset=ECG200
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import defaultdict
import matplotlib.pyplot as plt

# Keras data generator class for the ECG200 dataset
class ECG200DataGenerator(tf.keras.utils.Sequence):

  def __init__(self, dataset_filepath, batch_size):
    """
    Constructor Parameters;
    dataset_filepath:   the full path to a file containing data
    batch_size:         the batch size for the network
    
    """
    
    
    self.datasetFilepath = dataset_filepath

    #Load the dataset into memory
    self.data = np.loadtxt(dataset_filepath)

    #Seperate the data into time-series and the corresponding labels
    self.labels = self.data[:, 0]
    self.features = self.data[:, 1:]

    #The number of features for each row
    self.sequenceLength = self.features.shape[1]
    print(f"Time-Series Sequence Length: {self.sequenceLength}")

    #The batch size
    self.batchSize = batch_size

    # Return nothing

  def __len__(self):
    """
    Calculate the number of batches used for one epoch

    uses np.ciel so if the last batch size < given batch size, we still train on the remaining data
    """
    batches_per_epoch = int(np.ceil(len(self.labels) / self.batchSize))

    return batches_per_epoch


  def __getitem__(self, index):
    """
    Parameters:
    index:  the index of the batch to be retrieved

    Returns:
    x:      one batch of data
    y:      the one-hot encoded labels associated with the batch
    """
    lower = index * self.batchSize
    upper = (index + 1) * self.batchSize

    x = self.features[lower:upper]
    y = self.labels[lower:upper]

    # y is sequenceLength * 1
    # yOneHot is sequenceLength * 2
    yOneHot = np.zeros((len(y),2))
    yOneHot[np.arange(len(y)), (y == 1).astype(int)] = 1

    x = x.reshape(-1, self.sequenceLength, 1)

    return x, yOneHot

# A function that creates a keras cnn model to predict which class a time-series corresponds to
def trainCNN(trainingDataFilePath, validationDataFilePath, learningRate=0.01, batchSize=16, epoch=30):
  """
  Function used to create a convolutional neural network model used to predict, given a sequence of electrical activity recorded 
  during one heartbeat of a patient by electrocardiogram (ECG), whether the heartbeat was normal or had myocaridal infraction.

  Parameters:
  trainingDataFilePath:   the full path to a file containing the training data
  validationDataFilePath: the full path to a file containing the validation data
  learningRate:           the learning rate used for the optimizer
  batchSize:              the batch size used for the data generator
  epoch:                  the number of epochs to train the model for

  Returns:
  model:                  a trained tf.keras convolutional neural network model to predict which class a time-series corresponds to
  trainingPerformance:    the performance of the model on the training set
  validationPerformance:  the performance of the model on the validation set
  """

  #Create a data generator for the training and validation datasets
  trainingDataGenerator = ECG200DataGenerator(trainingDataFilePath, batchSize)
  validationDataGenerator = ECG200DataGenerator(validationDataFilePath, batchSize)

  def CNN(numFeatures, learningRate=0.01):
    """
    Convolutional Neural Network

    k: kernel size
    s: stride
    p: padding
    a: activation function

    Architecutre:
      Input:                      96, 1
      Convolution Layer 1:        filters:1->8, k=6, s=1, p=0, a=ReLU      
      Conovlution Layer 2:        filters:8->32, k=3, s=2, p=0, a=ReLU
      Max Pooling Layer:          poolSize=2, s=2, p=0
      Fully Connected 1 Layer:    300 neurons,  a=ReLU
      Fully Connected 2 Layer:    100 neurons,  a=ReLU
      Output Layer:               2 neurons,   a=softmax
    
    Hyper-Parameters:
      Learning Rate:  0.01 (default)
      Optimizer:      Adaptive Moment Estimation (Adam) Optimizer
      Loss:           Categorical Cross-Entropy Loss

    """
    #Model created using Keras Functional API
    #Layers
    inputLayer = tf.keras.Input(shape=(numFeatures,1))

    #Convolution Layers
    convLayer1 = tf.keras.layers.Conv1D(filters=8, kernel_size=6, strides=1, padding="valid", activation="relu")
    convLayer2 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding="valid", activation="relu")

    #Max Pooling Layer
    poolLayer = tf.keras.layers.MaxPool1D(pool_size=2,strides=2,padding="valid")

    #Flatten input to be inputted into fully-connected feedforward network 
    flattenLayer = tf.keras.layers.Flatten()

    denseLayer1 = tf.keras.layers.Dense(300, activation="relu")
    denseLayer2 = tf.keras.layers.Dense(100, activation="relu")

    outputLayer = tf.keras.layers.Dense(2, activation="softmax")

    #Forward Pass
    conv1 = convLayer1(inputLayer)
    conv2 = convLayer2(conv1)
    pool = poolLayer(conv2)

    flatten = flattenLayer(pool)

    dense1 = denseLayer1(flatten)
    dense2 = denseLayer2(dense1)

    output = outputLayer(dense2)

    #Create the model
    model = tf.keras.Model(inputs=inputLayer, outputs=output)

    #Adaptive Moment Estimation Optimizer with Learning Rate (default=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)

    # Compile the model with Adam and Categorical Cross-Entropy Loss
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print a summary of the model
    print(model.summary())
  
    return model
  
  model = CNN(trainingDataGenerator.sequenceLength, learningRate)

  model.fit(x=trainingDataGenerator, 
            validation_data=validationDataGenerator, 
            epochs=epoch, shuffle=True)

  trainingPerformance = model.evaluate(trainingDataGenerator)

  validationPerformance = model.evaluate(validationDataGenerator)
  
  return model, trainingPerformance, validationPerformance

# Function used to test a given model on a given testing dataset
def testModel(model, testingDataFilePath, batchSize=16):
  testDataGenerator = ECG200DataGenerator(testingDataFilePath, batchSize)

  testPerformance = model.evaluate(testDataGenerator)
  
  return testPerformance

#Function used to test the convolutional neural network
def testCNN():
  model, trainingPerformance, validationPerformance = trainCNN('ECG200_TRAIN.tsv','ECG200_VALIDATION.tsv')
  testPerformance = testModel(model, 'ECG200_TEST.tsv', 16)

  print(f"---------------------Training Performance---------------------\nAccuracy: {trainingPerformance[1]}\t|\tLoss: {trainingPerformance[0]}\n---------------------------------------------------------------")

  print(f"---------------------Validation Performance---------------------\nAccuracy: {validationPerformance[1]}\t|\tLoss: {validationPerformance[0]}\n---------------------------------------------------------------")

  print(f"---------------------Test Performance---------------------\nAccuracy: {testPerformance[1]}\t|\tLoss: {testPerformance[0]}\n---------------------------------------------------------------")

# A function that creates a keras rnn model to predict which class a time-series corresponds to
def trainRNN(trainingDataFilePath, validationDataFilePath, learningRate=0.01, batchSize=16, epoch=75):
  """
  Function used to create a convolutional neural network model used to predict, given a sequence of electrical activity recorded 
  during one heartbeat of a patient by electrocardiogram (ECG), whether the heartbeat was normal or had myocaridal infraction.

  Parameters:
  trainingDataFilePath:   the full path to a file containing the training data
  validationDataFilePath: the full path to a file containing the validation data
  learningRate:           the learning rate used for the optimizer
  batchSize:              the batch size used for the data generator
  epoch:                  the number of epochs to train the model for

  Returns:
  model:                  a trained tf.keras convolutional neural network model to predict which class a time-series corresponds to
  trainingPerformance:    the performance of the model on the training set
  validationPerformance:  the performance of the model on the validation set
  """

  #Create a data generator for training and validation datasets
  trainingDataGenerator = ECG200DataGenerator(trainingDataFilePath, batchSize)
  validationDataGenerator = ECG200DataGenerator(validationDataFilePath, batchSize)

  def RecurrentNeuralNetwork(sequenceLength, learningRate=0.01):
    """
    Recurrent Neural Network

    Architecutre:
      Input:                          96, 1
      Long Short-Term Memory Layer:   16 hidden states
      Fully Connected 1 Layer:        32 neurons,  a=tanh
      Fully Connected 2 Layer:        16 neurons,  a=tanh
      Output Layer:                   2 neurons,   a=softmax
    
    Hyper-Parameters:
      Learning Rate:  0.01 (default)
      Optimizer:      Root Mean Square Propogation (RMSProp) Optimizer
      Loss:           Categorical Cross-Entropy Loss
    """
    #Model built using Keras Functional API
    
    #Input layer of (96, 1) representing the 96 timesteps and 1 feature per timestep
    inputLayer = keras.Input(shape=(sequenceLength, 1))

    #Long Short-Term Memory
    lstmLayer = keras.layers.LSTM(16)
    
    #Fully-Connected Layers
    denseLayer = keras.layers.Dense(32, activation='tanh')
    denseLayer1 = keras.layers.Dense(16, activation='tanh')

    #Output layer has 2 neurons for multi-class classification
    #Softmax activation function
    outputLayer = keras.layers.Dense(2, activation='softmax')

    #Forward Pass
    lstm = lstmLayer(inputLayer)
    dense = denseLayer(lstm)
    dense1 = denseLayer1(dense)
    output = outputLayer(dense1)

    #Build Model
    model = keras.Model(inputs=inputLayer, outputs=output)

    
    #90% training, 75% validation, 75% testing with lr=0.01
    #Adaptive Moment Estimation (Adam) Optimizer
    # optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)

    #95% training, 85% validation, 82% testing with lr=0.01
    #Root Mean Square Error Propogation (RMSProp) Optimizer
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = learningRate)

    #Compile the model using RMSProp and Categorical Cross-Entropy Loss
    model.compile(
      optimizer = optimizer,
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )

    print(model.summary())

    #Return the model
    return model

  model = RecurrentNeuralNetwork(trainingDataGenerator.sequenceLength, learningRate)
  
  model.fit( x=trainingDataGenerator, 
            validation_data=validationDataGenerator, 
            epochs=epoch, shuffle=True)

  trainingPerformance = model.evaluate(trainingDataGenerator)
  validationPerformance = model.evaluate(validationDataGenerator)

  return model, trainingPerformance, validationPerformance

# Function used to test the recurrent neural network
def testRNN():
  model, trainingPerformance, validationPerformance = trainRNN('ECG200_TRAIN.tsv','ECG200_VALIDATION.tsv')
  testPerformance = testModel(model, 'ECG200_TEST.tsv', 16)
  print(f"---------------------Training Performance---------------------\nAccuracy: {trainingPerformance[1]}\t|\tLoss: {trainingPerformance[0]}\n---------------------------------------------------------------")

  print(f"---------------------Validation Performance---------------------\nAccuracy: {validationPerformance[1]}\t|\tLoss: {validationPerformance[0]}\n---------------------------------------------------------------")

  print(f"---------------------Test Performance---------------------\nAccuracy: {testPerformance[1]}\t|\tLoss: {testPerformance[0]}\n---------------------------------------------------------------")

# Function used to experiment on both convolutional neural network and recurrent neural network 
# Allows us to find the optimal parameters
def parameterTuning(trainingDataFilePath, validationDataFilePath):

  # trainingDataFilePath is the full path to a file containing the training data
  # validationDataFilePath is the full path to a file containing the validation data

  trainingDataGenerator = ECG200DataGenerator(trainingDataFilePath, 16)
  validationDataGenerator = ECG200DataGenerator(validationDataFilePath, 16)

  def CNN_experiment(numFeatures, learningRate, kernel):

    #Input of size (96, 1)
    inputLayer = tf.keras.Input(shape=(numFeatures,1))

    #Convolution
    convLayer1 = tf.keras.layers.Conv1D(filters=8, kernel_size=kernel, strides=1, padding="valid", activation="relu")
    convLayer2 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel, strides=2, padding="valid", activation="relu")

    #Max Pooling
    poolLayer = tf.keras.layers.MaxPool1D(pool_size=2,strides=2,padding="valid")

    #Flatten input to be input into fully connected feed forward network part
    flattenLayer = tf.keras.layers.Flatten()

    #Fully-Connected Layers
    denseLayer1 = tf.keras.layers.Dense(300, activation="relu")
    denseLayer2 = tf.keras.layers.Dense(100, activation="relu")

    outputLayer = tf.keras.layers.Dense(2, activation="softmax")

    #Forward Pass
    conv1 = convLayer1(inputLayer)
    conv2 = convLayer2(conv1)
    pool = poolLayer(conv2)

    flatten = flattenLayer(pool)

    dense1 = denseLayer1(flatten)
    dense2 = denseLayer2(dense1)

    output = outputLayer(dense2)

    #Create the model
    model = tf.keras.Model(inputs=inputLayer, outputs=output)

    #Adaptive Moment Estimation (Adam) Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)

    # Compile the model with Adam and Categorical Cross-Entropy Loss
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print a summary of the model
    print(model.summary())

    return model
  
  def RNN_experiment(sequenceLength, learningRate=0.001, numHidden=32, type='lstm'):
    
    #Input layer of (96, 1) representing the 96 timesteps and 1 feature per timestep
    inputLayer = keras.Input(shape=(sequenceLength, 1))

    # Recurrent Unit can either be 
    # Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU)
    if type == 'lstm':
      recurrentLayer = keras.layers.LSTM(numHidden)
    else:
      recurrentLayer = keras.layers.GRU(numHidden)

    #Fully-Connected Layers
    denseLayer = keras.layers.Dense(32, activation='tanh')
    denseLayer1 = keras.layers.Dense(16, activation='tanh')

    #Output layer has 2 neurons for multi-class classification
    #Softmax activation function
    outputLayer = keras.layers.Dense(2, activation='softmax')

    #Forward Pass
    lstm = recurrentLayer(inputLayer)
    dense = denseLayer(lstm)
    dense1 = denseLayer1(dense)
    output = outputLayer(dense1)
    
    #Build the model
    model = keras.Model(inputs=inputLayer, outputs=output)

    #Adaptive Moment Estimation (Adam) Optimizer
    # optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)

    #Root Mean Square Error Propogation (RMSProp) Optimizer
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = learningRate)

    #Compile the model using RMSProp and Categorical Cross-Entropy Loss
    model.compile(
      optimizer = optimizer,
      loss='categorical_crossentropy',
      metrics=['accuracy']
    )

    #Print a summary of the model
    print(model.summary())

    #Return the model
    return model
  
  types = ['lstm', 'gru']

  hiddenStates = [16, 32, 48, 64, 72, 96]

  kernelSizes = [2, 4, 6, 8]

  epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
  
  #Dictionary used to hold results for the CNN and RNN experiments
  cnnResults = defaultdict()
  rnnResults = defaultdict()

  #Convolutional Experiment
  for kernel in kernelSizes:

    for epoch in epochs:

      model = CNN_experiment(trainingDataGenerator.sequenceLength, learningRate=0.01, kernel=kernel)

      model.fit(x=trainingDataGenerator, 
                validation_data=validationDataGenerator, 
                epochs=epoch, shuffle=True)

      trainingPerformance = model.evaluate(trainingDataGenerator)
      validationPerformance = model.evaluate(validationDataGenerator)

      cnnResults[(kernel, epoch)] = (trainingPerformance, validationPerformance)

  print(cnnResults)
  
  #Recurrent Experiment
  for type in types:

    for hidden in hiddenStates:

      for epoch in epochs:

        model = RNN_experiment(trainingDataGenerator.sequenceLength, 0.01, hidden, type)

        model.fit( x=trainingDataGenerator, 
                  validation_data=validationDataGenerator, 
                  epochs=epoch, shuffle=True)


        trainingPerformance = model.evaluate(trainingDataGenerator)
        validationPerformance = model.evaluate(validationDataGenerator)

        rnnResults[(type, hidden, epoch)] = (trainingPerformance, validationPerformance)

  print(rnnResults)

  return cnnResults, rnnResults

# Function used to plot the results of experiments on recurrent neural network
def plotRecurrentResults(results):
  # for tuple in results:
  #   print(f"{tuple} : {results[tuple]}")

  types = ['lstm', 'gru']

  hiddenStates = [16, 32, 48, 64, 72, 96]

  epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

  # Colors for each hidden state
  colors = {
      16: 'red',
      32: 'blue',
      48: 'green',
      64: 'orange',
      72: 'cyan',
      96: 'purple'
  }

  for type in types:
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    #training
    # Accuracy Plot
    for hidden in hiddenStates:
        accuracies = [results[type,hidden,epoch][0][1] for epoch in epochs]
        ax1.plot(epochs, accuracies, marker='o', color=colors[hidden], label=f'Hidden={hidden}')
    ax1.set_title(f'{type} | Training Accuracy vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss Plot
    for hidden in hiddenStates:
        losses = [results[type,hidden,epoch][0][0] for epoch in epochs]
        ax2.plot(epochs, losses, marker='o', color=colors[hidden], label=f'Hidden={hidden}')
    ax2.set_title(f'{type} | Training Loss vs Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # validation
    # Accuracy Plot
    for hidden in hiddenStates:
        accuracies = [results[type,hidden,epoch][1][1] for epoch in epochs]
        ax3.plot(epochs, accuracies, marker='o', color=colors[hidden], label=f'Hidden={hidden}')
    ax3.set_title(f'{type} | Validation Accuracy vs Epochs')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True)

    # Loss Plot
    for hidden in hiddenStates:
        losses = [results[type,hidden,epoch][1][0] for epoch in epochs]
        ax4.plot(epochs, losses, marker='o', color=colors[hidden], label=f'Hidden={hidden}')
    ax4.set_title(f'{type} | Validation Loss vs Epochs')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)


    plt.tight_layout()
    plt.show()

# Function used to plot the results of the experiemtns on convolutional neural networks
def plotConvolutionalResults(results):

  # for tuple in results:
  #   print(f"{tuple} : {results[tuple]}")

  kernelSizes = [2, 4, 6 ,8]

  epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

  # Colors for each kernel size
  colors = {
      2: 'red',
      4: 'blue',
      6: 'green',
      8: 'orange',
  }

  # Plotting
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
  fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

  #training
  # Accuracy Plot
  for kernel in kernelSizes:
      accuracies = [results[kernel,epoch][0][1] for epoch in epochs]
      ax1.plot(epochs, accuracies, marker='o', color=colors[kernel], label=f'Kernel={kernel}')
  ax1.set_title('Training | Accuracy vs Epochs')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Accuracy')
  ax1.legend()
  ax1.grid(True)

  # Loss Plot
  for kernel in kernelSizes:
      losses = [results[kernel,epoch][0][0] for epoch in epochs]
      ax2.plot(epochs, losses, marker='o', color=colors[kernel], label=f'Kernel={kernel}')
  ax2.set_title('Training | Loss vs Epochs')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Loss')
  ax2.legend()
  ax2.grid(True)

  # validation
  # Accuracy Plot
  for kernel in kernelSizes:
      accuracies = [results[kernel,epoch][1][1] for epoch in epochs]
      ax3.plot(epochs, accuracies, marker='o', color=colors[kernel], label=f'Kernel={kernel}')
  ax3.set_title('Validation Accuracy vs Epochs')
  ax3.set_xlabel('Epochs')
  ax3.set_ylabel('Accuracy')
  ax3.legend()
  ax3.grid(True)

  # Loss Plot
  for kernel in kernelSizes:
      losses = [results[kernel,epoch][1][0] for epoch in epochs]
      ax4.plot(epochs, losses, marker='o', color=colors[kernel], label=f'Kernel={kernel}')
  ax4.set_title('Validation Loss vs Epochs')
  ax4.set_xlabel('Epochs')
  ax4.set_ylabel('Loss')
  ax4.legend()
  ax4.grid(True)


  plt.tight_layout()
  plt.show()


# end of file