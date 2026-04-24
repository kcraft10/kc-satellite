# data loading and pre-processing steps
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# loading the data files and seeing the shape of the dataset
def data_load():
    curr_dir=os.path.dirname(os.path.abspath(__file__))
    training_file = os.path.join(curr_dir, 'P-1train.npy')
    test_file = os.path.join(curr_dir, 'P-1test.npy')
    labeled_anom_file = os.path.join(curr_dir, 'labeled_anomalies.csv')

    training_data = np.load(training_file)
    test_data = np.load(test_file)
    labeled_anom = pd.read_csv(labeled_anom_file)

    X_train = pd.DataFrame(training_data)
    X_test = pd.DataFrame(test_data)

    print(f"Train set shape{X_train.shape}")
    print(f"Test set shape{X_test.shape}")

    #print(X_train.head())
    #print(X_test.head())

# creating a time step so we can plot and view the channel over time and
# adding labels columns starting with 0 to use for putting in anomaly/normal
# values
    X_train['step'] = X_train.index
    X_train['label'] = 0
    X_test['step'] = X_test.index
    X_test['label'] = 0

# getting the labels from the csv that are from P-1 channel to see where to
# highlight them in the graph to show actual anoms
    anom=labeled_anom[labeled_anom['chan_id']=='P-1']
    anom_list=anom['anomaly_sequences'].values
    print(f"Sequence of Labeled Anomalies: {anom_list}")
# now can assign those sequences of labeled anoms the value 1 so we can quantify
# how good the model is doing later
    X_test.loc[2149:2349, 'label']=1
    X_test.loc[4536:4844, 'label']=1
    X_test.loc[3539:3779, 'label']=1

    return X_train, X_test

#Plotting figure of anomalies and defining anomalies as where it is 1
def training_plot(train):
    plt.figure(figsize=(10,8))
    plt.plot(X_train['step'], X_train[0])
    plt.xlabel('Time Step (Index)')
    plt.ylabel('Normalized Telemetry Value')
    plt.title('SMAP P-1 Channel Telemetry (Training)')
    plt.show()

def testing_plot(test):
    plt.figure(figsize=(10,8))
    plt.plot(X_test['step'], X_test[0])
    anomalies=X_test[X_test['label']==1]
    plt.scatter(anomalies['step'], anomalies[0], color='red', label='Labeled Anomalies')
    plt.xlabel('Time Step (Index)')
    plt.ylabel('Normalized Telemetry Value')
    plt.title('SMAP P-1 Channel Telemetry (Test)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X_train, X_test = data_load()
    training_plot(X_train)
    testing_plot(X_test)