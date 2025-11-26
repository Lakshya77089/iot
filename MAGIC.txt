# Linear regression + L1 (Lasso) + L2 (Ridge)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import make_regression

# 1. Create synthetic regression dataset
X, y = make_regression(n_samples=200,
                       n_features=3,
                       noise=10.0,
                       random_state=42)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Linear Regression (no regularization)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# 4. Ridge Regression (L2)
ridge = Ridge(alpha=1.0)   # alpha = λ (regularization strength)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# 5. Lasso Regression (L1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# 6. Print results
def print_results(name, y_true, y_pred):
    print(f"\n{name}")
    print("-" * len(name))
    print("MSE :", mean_squared_error(y_true, y_pred))
    print("R^2 :", r2_score(y_true, y_pred))

print_results("Linear Regression", y_test, y_pred_lin)
print_results("Ridge Regression", y_test, y_pred_ridge)
print_results("Lasso Regression", y_test, y_pred_lasso)





---------------------------------------------------------------
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load binary classification dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 4. Predictions
y_pred = log_reg.predict(X_test)

# 5. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
--------------------------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_iris()
X, y = data.data, data.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 4. Predict
y_pred = knn.predict(X_test)

# 5. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
----------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Data
data = load_iris()
X, y = data.data, data.target

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Decision Tree (sklearn uses CART)
tree_clf = DecisionTreeClassifier(
    criterion='gini',    # 'entropy' also available
    max_depth=None,
    random_state=42
)
tree_clf.fit(X_train, y_train)

# 4. Predict
y_pred = tree_clf.predict(X_test)

# 5. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
-----------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Use only two classes for clarity (e.g., iris)
data = load_iris()
X = data.data[:, :2]  # take first two features
y = data.target

# For simplicity, binary classification (class 0 vs 1)
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)
y_pred_lin = svm_linear.predict(X_test)

# Non-linear SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_lin))
print("RBF SVM Accuracy   :", accuracy_score(y_test, y_pred_rbf))
----------------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
---------------------------------------------------------------------------------
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Synthetic 2D data
X, y_true = make_blobs(n_samples=300,
                       centers=3,
                       cluster_std=0.6,
                       random_state=0)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

print("Cluster centers:\n", kmeans.cluster_centers_)
print("First 10 cluster labels:", kmeans.labels_[:10])
-------------------------------------------------------------------------------------
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data = load_iris()
X = data.data

# Keep 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
---------------------------------------------------------------------------------------
from sklearn.decomposition import FastICA
from sklearn.datasets import load_iris

data = load_iris()
X = data.data

ica = FastICA(n_components=3, random_state=42)
X_ica = ica.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_ica.shape)
--------------------------------------------------------------------------------------
import numpy as np
import random

# Simple environment settings
n_states = 6         # states: 0..5
n_actions = 2        # actions: 0 (left), 1 (right)
terminal_state = 5

# Reward structure (toy)
rewards = np.zeros((n_states, n_actions))
rewards[4, 1] = 1.0   # from state 4 going right to 5 gives reward +1

# Q-table init
Q = np.zeros((n_states, n_actions))

alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.1 # exploration probability
n_episodes = 500

def step(state, action):
    """Toy transition: left = -1, right = +1 (with bounds)."""
    if state == terminal_state:
        return state, 0.0
    next_state = state + (1 if action == 1 else -1)
    next_state = max(0, min(terminal_state, next_state))
    reward = rewards[state, action]
    return next_state, reward

for episode in range(n_episodes):
    state = 0  # start at 0 each episode
    while state != terminal_state:
        # epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q[state, :])

        next_state, reward = step(state, action)

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state

print("Learned Q-table (Q-learning):")
print(Q)


