#doing lstm-vae to see if I can improve the metrics a little bit

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from satellite_if import clean_data
from data_proc import data_load

X_train, X_test = data_load()
steps_test, truth_test, X_train_feat, X_test_feat = clean_data(X_train, X_test)

#using same sliding window by flattening the data into one list so the model can compare
# the values with other values
# (Method from Kaggle "Sliding Window with Isolation Forest")
window_size=50
num_feat=25
total_feat = window_size * num_feat

# padding the time lag so input=output length rather than starting late
# creates a window for each row, flattens it, then fills in any missing with nan
def window(feat, window_size, total_feat):
    X_window = np.array([np.pad(feat.iloc[max(0,i-window_size+1):i+1].values.flatten(),
                                (total_feat-len(feat.iloc[max(0,i-window_size+1):i+1].values.flatten()),0),
                                mode='constant',constant_values=np.nan)
                         for i in range(len(feat))])
    return X_window
# then adding the imputer to fix the nan values
imputer = SimpleImputer(strategy='most_frequent')

X_train_window=window(X_train_feat,window_size, total_feat)
X_test_window=window(X_test_feat,window_size, total_feat)

X_train_window = imputer.fit_transform(X_train_window)
X_test_window = imputer.transform(X_test_window)

# Implementing VAE Model (Method from GeeksforGeeks "Variational Autoencoders")
# sampling layer which allows VAE to get varied outputs
class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var=inputs
        batch=tf.shape(mean)[0]
        dim=tf.shape(mean)[1]
        epsilon=tf.random.normal(shape=(batch, dim))
        return mean+tf.exp(0.5*log_var)*epsilon

# making encoder by defining dimensions from flattening
latent_dim=16
input_dimensions=X_train_window.shape[1]
X_train_shape=X_train_window.reshape(-1, 50, 25)
X_test_shape=X_test_window.reshape(-1, 50, 25)

encoder_inputs=keras.Input(shape=(50, 25))
x1=layers.LSTM(8, activation='sigmoid', return_sequences=False)(encoder_inputs)
x3=layers.Dropout(0.10)(x1)
mean=layers.Dense(latent_dim, name='mean')(x3)
log_var=layers.Dense(latent_dim, name='log_var')(x3)
z=Sampling()([mean, log_var])
encoder=keras.Model(encoder_inputs, [mean, log_var, z], name='encoder')
encoder.summary()

# making decoder
latent_inputs=keras.Input(shape=(latent_dim,))
x1=layers.RepeatVector(50)(latent_inputs)
x2=layers.LSTM(8, activation='sigmoid', return_sequences=True)(x1)
x4=layers.Dropout(0.10)(x2)
decoder_outputs=layers.TimeDistributed(layers.Dense(25, activation='sigmoid'))(x4)
decoder=keras.Model(latent_inputs, decoder_outputs, name='decoder')
decoder.summary()

# now getting VAE model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(data)
            reconstruction=self.decoder(z)
            #reconstruction_loss=tf.reduce_mean(tf.reduce_sum(tf.square(data-reconstruction), axis=1))
            mse=tf.reduce_mean(tf.square(data-reconstruction), axis=(1,2))
            reconstruction_loss=tf.reduce_mean(mse)

            #bce=keras.losses.binary_crossentropy(data, reconstruction)
            #reconstruction_loss=tf.reduce_mean(tf.reduce_sum(bce,axis=-1))

            kl_loss=-0.5*(1+log_var-tf.square(mean)-tf.exp(log_var))
            kl_loss=tf.reduce_mean(tf.reduce_sum(kl_loss,axis=1))
            total_loss=reconstruction_loss+kl_loss
        grads=tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {'loss': self.total_loss_tracker.result(), 'reconstruction_loss': self.reconstruction_loss_tracker.result(), 'kl_loss': self.kl_loss_tracker.result()}

# now training the VAE
vae=VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(X_train_shape, epochs=30, batch_size=1000)

# now can do the data reconstruction then compare metrics by finding the reconstruction
# loss like in the training fxn which is also the mse here
z_mean, _, _ =vae.encoder.predict(X_test_shape)
data_recon=vae.decoder.predict(z_mean)
lloss=np.mean((X_test_shape-data_recon)**2, axis=(1,2))
#bce_test=keras.metrics.binary_crossentropy(X_test_window, data_recon)
#loss=bce_test.numpy()
loss_threshold=np.percentile(lloss,85)

# getting performance metrics to compare to iso forest model
# getting y pred from how large mse is to predict as anom
#making them have the same number of rows since the window causes a mismatch
#truth_test_start=truth_test.iloc[window_size-1:].values
y_pred=np.where(lloss>loss_threshold, 1, 0)

length=min(len(truth_test), len(y_pred))
truth_test_start=truth_test.iloc[-length:].values
y_pred_final=y_pred[-length:]

conf_matrix=confusion_matrix(truth_test_start, y_pred_final)
print(f"Confusion Matrix: {conf_matrix}")

# plotting where the model predicts anomalies
def lstmvae(feature, time_step, predictions):
    plt.figure(figsize=(10,8))
    plt.plot(time_step, feature.iloc[:, 0], alpha=0.5)
    anomalies = time_step[predictions==1]
    values=feature.iloc[:,0][predictions==1]
    plt.scatter(anomalies, values,color='red', label = 'Predicted Anomalies', s=12)
    plt.title('LSTM-VAE Model Anomalies')
    plt.xlabel('Time Step (Index)')
    plt.ylabel('Normalized Telemetry Value')
    plt.legend()
    plt.show()

lstmvae(X_test_feat, steps_test, y_pred)

#metrics as well
print(f"Precision: {precision_score(truth_test_start, y_pred_final)}")
print(f"Recall: {recall_score(truth_test_start, y_pred_final)}")
print(f"F1: {f1_score(truth_test_start, y_pred_final)}")