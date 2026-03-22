from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

feature_X = data.data 
target_Y = data.target

print(feature_X.shape)
print(target_Y.shape)

#There are 569 samples in the dataset 

#The number of samples do not determine whether the class is balanced or imbalanced, to determine this we must know two things;
#Find all the classes in the target data 
#Determine an outlier count in the class 

#A balanced dataset means there would be marginal skewness in the prediction as each class has similar amount of samples to represent that class, avoiding a prediction bias 
#An imbalanced dataset means there is a majority class that is predicted each time 

unique, counts = np.unique(target_Y, return_counts=True) #returning all the classes in target_Y, returning the counts of target_Y

for unique_class, count in zip(unique, counts): #Pairing the elements of unique and counts 
    print(f"Class {unique_class}: {count}")

#The dataset is imbalanced, Class 1 is represented by 145 more samples, which means the classes are not equally represented 
#Class imbalance is important as this model could be biased and achieve a higher accuracy prediciting the majority class, however, do worse on the minority

