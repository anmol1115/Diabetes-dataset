import pandas as pd
X_train=pd.read_csv(r'C:\\Users\Anmol\Desktop\ML Masters\Diabetes_Xtrain.csv')
y_train=pd.read_csv(r'C:\\Users\Anmol\Desktop\ML Masters\Diabetes_ytrain.csv').iloc[:,0]
X_test=pd.read_csv(r'C:\\Users\Anmol\Desktop\ML Masters\Diabetes_Xtest.csv')

solver=['newton-cg', 'liblinear', 'sag', 'saga', 'lbfgs']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train_new,X_test_new,y_train_new,y_test_new=train_test_split(X_train,y_train,test_size=0.132,random_state=0)
best_solver=str()
score=0
for string in solver:
    classifier=LogisticRegression(solver=string,max_iter=150)
    classifier.fit(X_train_new,y_train_new)
    y_pred=classifier.predict(X_test_new)
    cn=confusion_matrix(y_true=y_test_new,y_pred=y_pred)
    score_cal=((cn[0,0]+cn[1,1])/(cn[0,0]+cn[0,1]+cn[1,0]+cn[1,1]))
    if score<score_cal:
        score=score_cal
        best_solver=string

classifier_logreg=LogisticRegression(solver=best_solver)
classifier_logreg.fit(X_train,y_train)
y_pred_logreg=classifier_logreg.predict(X_test)