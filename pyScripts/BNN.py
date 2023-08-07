#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # Import TensorFlow Probability
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### standard sci-py libraries
import matplotlib.pyplot as plt
import pandas as pd
import uproot ### to read, convert, inspect ROOT TTrees

from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml import analysis_utils
from hipe4ml import plot_utils

hdl_mc = TreeHandler("/home/himanshu/Sharma/hipe4ml_Files/XicSignal.root", "XicSignal")
hdl_data = TreeHandler("/home/himanshu/Sharma/hipe4ml_Files/XicBkg.root", "XicBkg")
hdl_bkg = hdl_data.apply_preselections("fM < 2.42 or fM > 2.5", inplace=False)
print('Size of samples \nData = ',len(hdl_bkg), ', MC = ', len(hdl_mc))
print('----------------------------------')



# Samples for training
fractionData=0.1 # this is the fraction of total data used in training 
hdl_bkg.shuffle_data_frame(frac=fractionData, inplace=True)
print('Size of samples for training \nData = ',len(hdl_bkg), ', MC = ', len(hdl_mc))


hdl_data.apply_preselections("fM > 2.42 and fM < 2.5", inplace=True)


cols_to_be_compared = ['fM','fPt','fCpa','fCpaXY','fCt','fEta','fPhi', 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fDecayLengthXYNormalised', 
                'fErrorDecayLength' , 'fErrorDecayLengthXY', 'fChi2PCA', 'fRSecondaryVertex', 'fCandidateSelFlag',
                'fNSigTpcPi0','fNSigTpcPi2','fNSigTpcPr0','fNSigTpcPr2','fNSigTpcKa1',
                'fNSigTofPi0','fNSigTofPi2','fNSigTofPr0','fNSigTofPr2','fNSigTofKa1']


plot_utils.plot_distr([hdl_mc, hdl_bkg], cols_to_be_compared, 
                      bins=30, labels=['Signal', 'Background'],
                      log=1, density=True, figsize=(20, 20), alpha=0.3, grid=False);


# training_cols = ['fCpa','fCpaXY','fCt','fEta','fPhi', 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised','fNSigTpcPi0','fNSigTpcPi2','fNSigTpcPr0','fNSigTpcPr2','fNSigTpcKa1',
#                 'fNSigTofPi0','fNSigTofPi2','fNSigTofPr0','fNSigTofPr2','fNSigTofKa1', 'fErrorDecayLength' , 'fErrorDecayLengthXY', 'fChi2PCA', 'fRSecondaryVertex']

# training_cols = ['fCpa','fCt','fDecayLength', 'fNSigTpcPi0','fNSigTpcPi2','fNSigTpcPr0','fNSigTpcPr2','fNSigTpcKa1','fErrorDecayLengthXY', 'fChi2PCA', 'fRSecondaryVertex']
# training_cols = ['fCpa','fCt','fDecayLength', 'fNSigTofPi0','fNSigTofPi2','fNSigTofPr0','fNSigTofPr2','fNSigTofKa1','fErrorDecayLengthXY', 'fChi2PCA', 'fRSecondaryVertex']
training_cols = ['fCpa','fCpaXY',
                 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fDecayLengthXYNormalised',
                 'fCt',
#                  'fNSigTofPi0','fNSigTofPi2','fNSigTofPr0','fNSigTofPr2','fNSigTofKa1',
#                  'fNSigTpcPi0','fNSigTpcPi2','fNSigTpcPr0','fNSigTpcPr2','fNSigTpcKa1',
                 'fChi2PCA',
                 'fRSecondaryVertex'
                 
                ]
# train_test_data is a combined dataset of data and MC
train_test_data = analysis_utils.train_test_generator([hdl_bkg, hdl_mc], [0, 1], test_size=0.5, random_state=42)

print('Size of train sample: ', len(train_test_data[0]))
print('Size of test sample: ', len(train_test_data[3]))


# Define a Bayesian Neural Network model using TensorFlow Probability
def build_bnn_model():
    model = tf.keras.Sequential([
        tfp.layers.DenseFlipout(16, activation='relu'),
        tfp.layers.DenseFlipout(16, activation='relu'),
        tfp.layers.DenseFlipout(1, activation='sigmoid')
    ])
    return model


# Create a custom scikit-learn estimator using TensorFlow and TFP
from sklearn.base import BaseEstimator, ClassifierMixin

class BayesianNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs=100, batch_size=32):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = build_bnn_model()
    
    def fit(self, X, y):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        self.model.compile(loss=self.negative_log_likelihood, optimizer='adam')
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=0)
        return self
    
    def predict(self, X):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y_pred = self.model(X).numpy()
        return (y_pred > 0.5).astype(int)
    
    def negative_log_likelihood(self, y_true, y_pred):
        return -tf.reduce_mean(self.log_likelihood(y_true, y_pred))
    
    def log_likelihood(self, y_true, y_pred):
        return tf.reduce_sum(
            tfp.distributions.Bernoulli(logits=y_pred).log_prob(y_true)
        )

bnn = BayesianNeuralNetwork(n_epochs=500)

model_hdl = ModelHandler(bnn, training_cols)
# bnn.fit(train_test_data[0], train_test_data[1])

score_test = bnn.predict(train_test_data[2])

plt.hist(score_test, bins=100, label='full sample', alpha=0.5, density=True);
plt.legend();

plt.hist(score_test[train_test_data[3]==0], bins=100, label='background', alpha=0.5, density=True);
plt.hist(score_test[train_test_data[3]==1], bins=100, label='signal', alpha=0.5, density=True);
plt.legend();


plot_utils.plot_roc(train_test_data[3], score_test);

score_train = bnn.predict(train_test_data[0])
plot_utils.plot_roc(train_test_data[1], score_train);

plot_utils.plot_output_train_test(bnn, train_test_data, density=True, bins=100, logscale=True);

plot_utils.plot_feature_imp(train_test_data[2], train_test_data[3], model_hdl) 
