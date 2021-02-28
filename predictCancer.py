import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import pandas as pd

cancer= datasets.load_breast_cancer()

# print(list(cancer.feature_names))
# print(cancer.target_names)

x=cancer.data
y=cancer.target
print(pd.DataFrame(x).head())

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)
classes = ['malignant', 'benign']

model=svm.SVC(kernel="linear")
model.fit(x_train,y_train)
predictions= model.predict(x_test)
acc=metrics.accuracy_score(y_test,predictions)

print("Accuracy:",acc)
for i in range(len(predictions)):
    print("Actual result:",classes[y_test[i]],"Predicted result:",classes[predictions[i]])
