import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Genres
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load features and labels
fileName = '../../features/train-features.txt'
with open(fileName, 'r') as f:
    X = f.readlines()
X = [x.strip().split(',') for x in X]
y = np.array([genres.index(x.pop()) for x in X])
X = np.array(X, dtype=float)

# train test split
#X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train=X
#y_train=y

# preprocessing phase
scaler = StandardScaler()
scaler.fit(X_train)
# StandardScaler(copy=True, with_mean=True, with_std=True)

# apply the transformations to the data:
X_train = scaler.transform(X_train)

# training the model
# the nth of the tuple consisting of the number of neurons in the nth layer
# the cardinality of the tuple represents the number of layer

# build
mlp = MLPClassifier(hidden_layer_sizes=(400, 400, 400), max_iter=10000)
# train
mlp.fit(X_train, y)

'''
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
'''
# Save model using pickle
modelFileName = 'multi-layer-perceptron-model.pkl'
with open(modelFileName, 'wb') as f:
    pickle.dump(mlp, f)

#predictions = mlp.predict(X_test)
#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))

print("end")
