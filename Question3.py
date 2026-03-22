from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import pandas as pd

bc_data = load_breast_cancer()

feature_X = bc_data.data 
target_Y = bc_data.target

X_train, X_test, Y_train, Y_test = train_test_split(feature_X,target_Y,test_size=0.2, random_state= 42)

decision_Tree_Classifier = DecisionTreeClassifier(criterion= 'entropy', max_depth = 10)
decision_Tree_Classifier.fit(X_train, Y_train)

predicted_labels_test = decision_Tree_Classifier.predict(X_test)
accuracy_score_test = accuracy_score(Y_test, predicted_labels_test)

print(accuracy_score_test)  

#Displaying the most important features, we can use model.feature_importances_
importance = pd.Series(decision_Tree_Classifier.feature_importances_, index = bc_data.feature_names)
print(importance.sort_values(ascending=False))

#We see that the top 5 are, mean concave points, worst perimeter, worst texture, worst radius, mean texture)

#Controlling the model complexity affect overfitting as a more complex model is more likely to overfit, while a more generalized model is more likely to underfit
#We can explain this by using the bias-variance tradeoff, a model that is more complex, has a lower bias yet a greater variance, while a model that is less complex has a higher bias yet a lower variance
#We can use constraints in order to limit the variance trained in a model and avoid overfitting 

#Feature importance contributes to the interpretability of Decision Trees as it calculates the features that had the purest separations, or a large entropy reduction
#Therefore, we can see the features that produce the purest separations of entropy of the Decision Tree and therefore contribute more the Decision Trees prediction

