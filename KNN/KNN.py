# Ejercicio 03: Código kNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Definir modelo kNN con k=5
knn = KNeighborsClassifier(n_neighbors=5)

# 4. Entrenar
knn.fit(X_train, y_train)

# 5. Evaluar
y_pred = knn.predict(X_test)

print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))