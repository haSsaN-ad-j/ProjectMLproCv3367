import matplotlib.pyplot as plt
import pandas as pd
import numpy as num
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import RocCurveDisplay
from joblib import load,dump
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

surv = pd.read_csv(r"C:\Users\Admin\Downloads\train.csv")
surv_test =pd.read_csv(r"C:\Users\Admin\Downloads\test.csv")

surv['Title'] = surv['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
age_bins = [0, 12, 19, 59, 120]
age_labels = ['Child', 'Teenager', 'Adult', 'Senior']
surv['AgeGroup'] = pd.cut(surv['Age'], bins=age_bins, labels=age_labels)
age =['Age']
agepro = Pipeline([
    ("imputer",SimpleImputer(strategy="median", fill_value="missing"))
])

surv_test['Title'] = surv_test['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
surv_test['AgeGroup'] = pd.cut(surv_test['Age'], bins=age_bins, labels=age_labels)


cat_feature = ['Sex','Embarked','Title','AgeGroup']
cat_feature_pre = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent", fill_value="missing")),
    ("Hotencoder",OneHotEncoder(handle_unknown="ignore"))
])

num_features = ['Age', 'Fare', 'SibSp', 'Parch']
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    
])

x=surv.drop('Survived',axis=1)
y=surv["Survived"]

ctr = ColumnTransformer(
    transformers=[
        ("age",agepro,age),
        ("cat",cat_feature_pre,cat_feature),
        ("num",num_transformer,num_features)
    ]
)

model1 =Pipeline([
    ("ctr",ctr),
    ("model",LogisticRegression())
])

model2 =Pipeline([
    ("ctr",ctr),
    ("model",RandomForestClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)

print("Train accuracy model1:", model1.score(X_train, y_train))
print("Validation accuracy model1:", model1.score(X_test, y_test))
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))

print("Train accuracy model2:", model2.score(X_train, y_train))
print("Validation accuracy model2:", model2.score(X_test, y_test))
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

gender_data = surv["Sex"]
survival = surv["Survived"]

eda = pd.DataFrame({
    "sex": gender_data.head(20),
    "Survived": survival.head(20)
})


counts = eda.groupby(['sex', 'Survived']).size().unstack()

ax = counts.plot(kind='bar', stacked=False)
ax.set_title('Survival count by Sex (first 20 rows)')
ax.set_xlabel('Sex')
ax.set_ylabel('Count')

plt.show()

