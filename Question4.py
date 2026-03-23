from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

bc_data = load_breast_cancer()

feature_X = bc_data.data 
target_Y = bc_data.target

X_train, X_test, y_train, y_test = train_test_split(feature_X,target_Y, test_size = 0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='logistic',
    max_iter=500,
    random_state=42
)

nn_model.fit(X_train_scaled, y_train)

y_train_pred = nn_model.predict(X_train_scaled)
y_test_pred = nn_model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Neural networks train using gradient-based optimization.
# If one feature has much larger values than another, it can dominate the updates, causing slow or unstable learning. Standardizing puts features on a similar scale, which helps the network converge faster and more effectively.

# An epoch is one complete pass through the entire training dataset during learning.
# During each epoch, the neural network updates its weights to reduce prediction error.