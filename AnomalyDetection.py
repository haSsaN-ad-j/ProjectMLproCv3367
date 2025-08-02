import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


data = pd.read_csv(r"C:\Users\Admin\Downloads\creditcard.csv\creditcard.csv")
cl = data['Class']
data.drop('Class', axis=1, inplace=True)
print(data['Amount'].describe()) #max>25000,min=0,avg=88(too high and to low values -> nedd scaling )
print(data.isna().sum())

#plt.figure(figsize=(8,5))
#sns.histplot(data['Amount'],kde=True) #elft skewed
#plt.show()

#plt.figure(figsize=(8,5))
#sns.boxplot(data['Amount']) 
#plt.show()

#plt.figure(figsize=(8,5))
#sns.boxplot(data['Time']) 
#plt.show()

scaler =RobustScaler()
data['Amount']=scaler.fit_transform(data[['Amount']])

mms =MinMaxScaler()
data['Time']=mms.fit_transform(data[['Time']])
pca =PCA(n_components=0.95)
data = pca.fit_transform(data)
isolationforest =IsolationForest(
    n_estimators=200,            # number of trees
    contamination=0.001,          # expected % of frauds
    max_samples=1000,          # number of samples per tree
    random_state=42
)
isolationforest.fit(data)
predictions = isolationforest.predict(data)


predictions_converted = [1 if x == -1 else 0 for x in predictions]



precisionscore=precision_score(cl, predictions_converted)  
reccalscore=recall_score(cl, predictions_converted)     
f1score=f1_score(cl, predictions_converted)         

print(f"precision_score = {precisionscore}")
print(f"recall_score = {reccalscore}")
print(f"f1_score = {f1score}")

pca_df = pd.DataFrame(data, columns=[f'PC{i+1}' for i in range(data.shape[1])])
pca_df['TrueLabel'] = cl.values
pca_df['PredictedLabel'] = predictions_converted

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='TrueLabel',
    data=pca_df,
    palette={0: 'blue', 1: 'red'},
    alpha=0.4,
    s=10
)
plt.title('True Labels: Normal (blue) vs Fraud (red)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Class')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='PredictedLabel',
    data=pca_df,
    palette={0: 'green', 1: 'orange'},
    alpha=0.4,
    s=10
)
plt.title('Model Predictions: Normal (green) vs Anomaly (orange)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Prediction')
plt.show()


