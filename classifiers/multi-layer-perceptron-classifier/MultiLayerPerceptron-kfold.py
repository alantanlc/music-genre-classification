import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Genres
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load features and labels
trainFileName = '../../features/train-features.txt'
with open(trainFileName, 'r') as f:
    X = f.readlines()
X = [x.strip().split(',') for x in X]
y = np.array([genres.index(x.pop()) for x in X])
X = np.array(X, dtype=float)

# Load model using pickle
modelFileName = 'multi-layer-perceptron-model.pkl'
with open(modelFileName, 'rb') as f:
    mlp = pickle.load(f)

# pre processing
scaler = StandardScaler()
scaler.fit(X)
# apply the transformations to the data:
X = scaler.transform(X)
# KFold
kf = KFold(n_splits=10, shuffle=True)
result = []
rates = []
for train_index, test_index in kf.split(X):
    tp_rate = []
    # Split data to train and test set
    # print("TRAIN: \t\t" + str(train_index))
    # print("TEST: \t\t" + str(test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train
    mlp.fit(X_train, y_train)
    # Print accuracy
    print("Accuracy: \t" + str(mlp.score(X_test, y_test)))
    result.append(mlp.score(X_test, y_test))

    # I make the predictions
    predicted = mlp.predict(X_test)

    # I obtain the confusion matrix
    cm = confusion_matrix(y_test, predicted)
    # rate calculation
    i = 0
    for row in cm:
        current = 0
        TP = 0
        FP = 0
        for g in row:
            if current == i:
                TP = g
            else:
                FP = FP + g
            current = current + 1
        tp_rate.append(TP / (TP + FP))
        i = i + 1
    rates.append(tp_rate)

rates = np.round(np.mean(rates, axis=0),3)
print("")
print("accuracy mean:", np.mean(result))
i=0
for r in rates:
   print(genres[i],r)
   i=i+1


