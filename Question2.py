from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

bc_data = load_breast_cancer()

feature_X = bc_data.data 
target_Y = bc_data.target

X_train, X_test, Y_train, Y_test = train_test_split(feature_X,target_Y,test_size=0.2, random_state= 42)

decision_Tree_Classifier = DecisionTreeClassifier(criterion= 'entropy')
decision_Tree_Classifier.fit(X_train, Y_train)

predicted_labels_test = decision_Tree_Classifier.predict(X_test)
accuracy_score_test = accuracy_score(Y_test, predicted_labels_test)

predicted_labels_train = decision_Tree_Classifier.predict(X_train)
accuracy_score_train = accuracy_score(Y_train, predicted_labels_train)

print(accuracy_score_train) #1.00
print(accuracy_score_test)  #0.95

#We see that the accuracy score of the training set is 1.00, this makes sense as the model is trained on that data
#We see that the accuracy score of the testing set is 0.95, this makes sense as the training set performs well on testing set, but not perfectly, or too low which suggest underfitting

#Entropy represents the purity/or impurity of a dataset. A high entropy is when a node is perfectly uncertain (1), while a low entropy is when a node is perfectly certain (0).
# -This means that dataset with samples only from one class has low entropy, and dataset that is perfectly balanced has a high entropy
# -In determining branches the lower entropy is assumed (closer to 0, more pure)

#This suggests that there is good generalization as the accuracy of the training set and testing set are close to eachother, the high training accuracy suggests that the model was trained and optimized on the training data, this does not mean the model is overfitting, and that given the testing set, it performs within 5% of that training accuracy
#Therefore, we can conclude that the model generalizes well to the testing set. Overfitting would require the testing set to be significantly lower to the training accuracy regardless of whether the training accuracy was perfect or not.
