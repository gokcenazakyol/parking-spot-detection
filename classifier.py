import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

# prepare data

path = "/Users/gokcenazakyol/Desktop/parking-spot-detection/clf-data"
categories = ["empty", "not_empty"]

data = []
labels = []

for category_index, category in enumerate(categories):
    for file in os.listdir(os.path.join(path, category)):
        image = imread(os.path.join(path, category, file))
        image = resize(image, (15, 15))
        data.append(image.flatten())
        labels.append(category_index)

data = np.asarray(data)
labels = np.asarray(labels)

# split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train model
clf = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(clf, parameters, cv=5)

grid_search.fit(X_train, y_train)

# evaluate model
best_model = grid_search.best_estimator_
y_predict = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy: ", accuracy)


# save model
pickle.dump(best_model, open("model.p", "wb"))