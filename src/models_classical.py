
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_knn(X_train, y_train):
    """
    Trains a K-Nearest Neighbors classifier with Grid Search for 'n_neighbors'.
    """
    print("Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'metric': ['euclidean', 'manhattan']}
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best KNN Params: {grid.best_params_}")
    return grid.best_estimator_

def train_dt(X_train, y_train):
    """
    Trains a Decision Tree Classifier using Gini index with Grid Search.
    """
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(criterion='gini', random_state=42)
    param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    grid = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best Decision Tree Params: {grid.best_params_}")
    return grid.best_estimator_

def train_svm(X_train, y_train):
    """
    Trains a Support Vector Machine with Linear and RBF kernels using Grid Search.
    """
    print("Training SVM...")
    svm = SVC(probability=True, random_state=42)
    param_grid = {
        'C': [0.1, 1, 10, 100], 
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best SVM Params: {grid.best_params_}")
    return grid.best_estimator_
