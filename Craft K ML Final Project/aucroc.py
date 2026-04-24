#doing a precision recall curve to compare the performances

from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from satellite_if import clean_data
from satellite_if import iso_forest, X_test_feat
from vae import loss, X_test_window
from lstm import lloss
from data_proc import data_load

X_train, X_test = data_load()
steps_test, truth_test, X_train_feat, X_test_feat = clean_data(X_train, X_test)

if_score=iso_forest.score_samples(X_test_feat)
vae_score=loss
lstm_score=lloss

precision_if, recall_if, threshold =precision_recall_curve(truth_test, if_score)
precision_vae, recall_vae, threshold =precision_recall_curve(truth_test, vae_score)
precision_lstm, recall_lstm, threshold=precision_recall_curve(truth_test, lstm_score)

plt.figure(figsize=(8,6))
plt.plot(recall_if, precision_if, label="IF score")
plt.plot(recall_vae, precision_vae, label="VAE score")
plt.plot(recall_lstm, precision_lstm, label="LSTM-VAE score")
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()