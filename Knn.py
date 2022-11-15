import numpy as np
import pandas as pd



#Load Dataset
dataset = pd.read_csv("dataset_path")


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Training and Predictions
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



import itertools

import matplotlib.pyplot as plt

importances_values = rf.feature_importances_
importances = pd.Series(importances_values, index=x_train.columns)
top20 = importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8, 6))
plt.title('Feature importances Top 20')
sns.barplot(x = top20, y = top20.index)
plt.show()

fs = RandomForestClassifier(max_features = 20, n_estimators=20, random_state=41)
fs.fit(X_train, y_train)
ranking = dict(zip(snp, fs.feature_importances_))
ranking = dict(itertools.islice(ranking.items(), n_features))
top_snp = list(ranking.keys())
top_snp = np.asarray(top_snp)

X_train = data.drop(test_index)
X_train = X_train[top_snp].values
X_test = data.reindex(test_index)
X_test = X_test[top_snp].values

tmp = knn(X_train, y_train, X_test)


# reference : https://nittaku.tistory.com/286
# reference : https://jaaamj.tistory.com/35
