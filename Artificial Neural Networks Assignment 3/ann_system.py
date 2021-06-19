import json
from numpy import loadtxt
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
import numpy as np

# Method to define, evaluate and return a model
def trainModel():
  dataset = loadtxt("training_dataset.csv", delimiter=",")

  X = dataset[:,0:8] # Splits the dataset to get the inputs
  Y = dataset[:,8] # Splits the dataset to get the output

  model = Sequential()
  model.add(Dense(12, input_dim=8, activation='relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

  # Test for accuracy of the model
  scores = model.evaluate(X, Y, verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

  # model.save("model.h5")

  return model

# Function to write the model to an external
# file that can be loaded in the future
def saveModel(model):
  model.save('model.h5')

# Function to load model from an external file
# and evaluate the accuracy of the model using test data
def loadModelAndEvaluate():
  model = load_model('model.h5')
  
  model.summary()
  dataset = loadtxt("test_data.csv", delimiter=",")
  
  X = dataset[:,0:8] # Splits the dataset to get the inputs
  Y = dataset[:,8] # Splits the dataset to get the output
  
  score = model.evaluate(X, Y, verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


# Function to load model from an external file
# and make prediction with the model using test data
def loadModelAndPredict():
  model = load_model('model.h5')
  
  model.summary()
  dataset = loadtxt("test_data.csv", delimiter=",")

  X = dataset[:,0:8]
  Y = dataset[:,8]

  # make class predictions with the model

  predictions = (model.predict(X) > 0.5).astype("int32")

  resultsJson = []

  for ndx in range(len(predictions)):
    resultsJson.append({
      "inputs": X[ndx].tolist(),
      "Expected Output": Y[ndx],
      "Actual Output": predictions[ndx] 
    })

  resultsJson = json.dumps(resultsJson, cls=NumpyEncoder)

  with open('results.json', 'w') as json_file:
    json_file.write(resultsJson)

  for ndx in range(len(predictions)):
    print('Inputs => %s \n Actual Ouput => %d \n Expected Output => %d' % (X[ndx].tolist(), predictions[ndx], Y[ndx]))

# This encoder will allow us to export our results to a json file
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)  

model = trainModel()
saveModel(model)
loadModelAndEvaluate()
loadModelAndPredict()