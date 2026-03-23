from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

bc_data = load_breast_cancer()

feature_X = bc_data.data 
target_Y = bc_data.target


X_train, X_test, y_train, y_test = train_test_split(feature_X,target_Y, test_size = 0.2, random_state = 42)

#Neural 

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

y_test_pred_nn = nn_model.predict(X_test_scaled) 

#Decision Tree 

decision_Tree_Classifier = DecisionTreeClassifier(criterion= 'entropy', max_depth = 10)
decision_Tree_Classifier.fit(X_train, y_train)

y_test_pred_tree = decision_Tree_Classifier.predict(X_test)

cm_dec_tree = confusion_matrix(y_test, y_test_pred_tree)
cm_nn = confusion_matrix(y_test, y_test_pred_nn)


print(cm_dec_tree)
print()
print(cm_nn)

#I would prefer the Neural Network since it achieves better test accuracy and a stronger confusion matrix.

#Constrained Decision Tree advantage:
#The constrained tree is easier to interpret, and adding limits such as max_depth
#or min_samples_split helps reduce overfitting compared to a fully grown tree.

#Constrained Decision Tree limitation:
#Even with constraints, the tree may still be too simple and miss important relationships in the data, especially if the pattern is more complex.

#Neural Network advantage:
#After feature scaling, the neural network can learn more effectively because all input features are on a similar scale. 
#Which improves optimization and helps it model complex nonlinear patterns.

#Neural Network limitation:
#The neural network is harder to interpret than a decision tree, and it depends on proper feature scaling and parameter tuning to perform well.