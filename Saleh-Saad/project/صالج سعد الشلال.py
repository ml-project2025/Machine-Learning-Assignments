import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = datasets.load_digits()

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(digits.images[i])
    plt.title(f'Label: {digits.target[i]}')
plt.show()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title(f"Prediction: {model.predict([digits.data[0]])[0]}")
plt.show()