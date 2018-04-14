import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix


wine = pd.read_csv('wine_data.csv',
                   names=["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium",
                          "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity",
                          "Hue", "OD280", "Proline"])

a = wine.head()
b = wine.describe().transpose()

# set up data and labels
X = wine.drop('Cultivator', axis=1)
y = wine['Cultivator']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# preprocessing phase
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
# apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# training the model
# the nth of the tuple consisting of the number of neurons in the nth layer
# the cardinality of the tuple represents the number of layer
#build
mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
#train
mlp.fit(X_train,y_train)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print("end")
