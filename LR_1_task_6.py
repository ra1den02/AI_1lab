import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Завантажте дані з файлу
data = np.loadtxt('data_multivar_nb.txt', delimiter=',')

# Визначте ознаки (X) та мітки класів (y)
X = data[:, :-1]
y = data[:, -1]

# Розбийте дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створіть та навчіть модель SVM
svm_model = SVC(kernel='linear')  # Ви можете вибрати інший тип ядра
svm_model.fit(X_train, y_train)

# Зробіть прогнози для SVM
svm_predictions = svm_model.predict(X_test)

# Виведіть показники якості для SVM
print("Support Vector Machine (SVM) Classification Report:")
print(classification_report(y_test, svm_predictions))
print("Accuracy:", accuracy_score(y_test, svm_predictions))

# Створіть та навчіть модель наївного байєсівського класифікатора
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Зробіть прогнози для наївного байєсівського класифікатора
nb_predictions = nb_model.predict(X_test)

# Виведіть показники якості для наївного байєсівського класифікатора
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))
print("Accuracy:", accuracy_score(y_test, nb_predictions))
