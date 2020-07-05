import pandas as pd
import numpy as np
X_train=pd.read_csv(r'C:\\Users\Anmol\Desktop\ML Masters\Diabetes_Xtrain.csv')
y_train=pd.read_csv(r'C:\\Users\Anmol\Desktop\ML Masters\Diabetes_ytrain.csv').iloc[:,0]
X_test=pd.read_csv(r'C:\\Users\Anmol\Desktop\ML Masters\Diabetes_Xtest.csv')

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train_new,X_test_new,y_train_new,y_test_new=train_test_split(X_train,y_train,test_size=0.132,random_state=0)
score=0
ideal_neigbors=0
for i in np.arange(1,10):
    classifier=KNeighborsClassifier(n_neighbors=i,metric='minkowski')
    classifier.fit(X_train_new,y_train_new)
    y_pred=classifier.predict(X_test_new)
    cn=confusion_matrix(y_true=y_test_new,y_pred=y_pred)
    score_cal=((cn[0,0]+cn[1,1])/(cn[0,0]+cn[0,1]+cn[1,0]+cn[1,1]))
    if score<score_cal:
        score=score_cal
        ideal_neigbors=i

classifier_knn=KNeighborsClassifier(n_neighbors=ideal_neigbors,metric='minkowski')
classifier_knn.fit(X_train,y_train)
y_pred_knn=classifier_knn.predict(X_test)