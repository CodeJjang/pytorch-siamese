from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, k):
        self.k = k

    def __call__(self, x_train, y_train, x_test, y_test):
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        neigh.fit(x_train, y_train)
        y_knn = neigh.predict(x_test)
        correct = (y_knn == y_test).astype(float).sum()
        acc = correct / len(y_test)
        return correct, acc
