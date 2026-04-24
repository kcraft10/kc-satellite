import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_proc import data_load

X_train, X_test = data_load()

#dropping the step and label columns so when the model is tested,
# it doesn't see the real labels from data_proc step
def clean_data(X_train, X_test):
    steps_test = X_test['step']
    truth_test = X_test['label']
    X_train_feat = X_train.drop(columns=['step', 'label'])
    X_test_feat = X_test.drop(columns=['step', 'label'])
    return steps_test, truth_test, X_train_feat, X_test_feat

steps_test, truth_test, X_train_feat, X_test_feat = clean_data(X_train, X_test)

#Isolation Forest algorithm
#(Methods from GeeksforGeeks example "Anomaly detection using Isolation Forest")
#defining the model with n trees, contamination which is proportion of anomalies,
# and then training the model to isolate the rows that seem odd
iso_forest = IsolationForest(n_estimators=1000, contamination = 0.08, random_state=20)
iso_forest.fit(X_train_feat)

#making predictions based on whether the model thinks a point is an anomaly or not
y_pred_train = iso_forest.predict(X_train_feat)
y_pred_test = iso_forest.predict(X_test_feat)

#printing which points showed red to compare with the ground truth
print(f"Locations of anomalies found (IF Model): {X_test_feat.index[y_pred_test == -1].values}")

# plotting test data and found anomalies from model
def iso_for_model(data):
    plt.figure(figsize=(10,8))
    plt.plot(data['step'], data[0], alpha=0.6)
    anomalies = data.index[y_pred_test==-1]
    plt.scatter(data.loc[anomalies,'step'], data.loc[anomalies, 0],color='red', label = 'Predicted Anomalies', s=18)
    plt.title('Isolation Forest Model Anomalies')
    plt.xlabel('Time Step (Index)')
    plt.ylabel('Normalized Telemetry Value')
    plt.legend()
    plt.show()

iso_for_model(X_test)

#confusion matrix to see how well model did
y_truth=np.where(y_pred_test==-1,1,0)
conf_matrix=confusion_matrix(truth_test, y_truth)
print(f"Confusion Matrix: {conf_matrix}")

#metrics
print(f"Precision: {precision_score(truth_test, y_truth)}")
print(f"Recall: {recall_score(truth_test, y_truth)}")
print(f"F1: {f1_score(truth_test, y_truth)}")