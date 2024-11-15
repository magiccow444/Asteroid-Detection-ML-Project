import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('nasa.csv')

df['Hazardous'] = df['Hazardous'].map({True: 1, False: 0})

y = df['Hazardous']

cat_cols = df.select_dtypes(include='category').columns
df = df.drop(cat_cols, axis=1)
X = df.drop(columns=['Hazardous', 'Close Approach Date', 'Orbiting Body', 'Orbit Determination Date', 'Equinox'])

pca = PCA(n_components=2)
X = pca.fit_transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

print(X_train.shape)
print(y_train.shape)

coefficients = np.polyfit(X_train[:, 0],  X_train[:, 1], 1)
poly = np.poly1d(coefficients)

plt.scatter(X_train[:, 0], X_train[:, 1])
plt.plot(poly, color='red')
plt.show()



