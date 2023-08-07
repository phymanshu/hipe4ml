#!/usr/bin/env python

### standard sci-py libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot ### to read, convert, inspect ROOT TTrees

import sys
print(sys.path)

mc_file = uproot.open("/home/himanshu/Sharma/hipe4ml_Files/XicSignal.root")
mc_file.keys()
# pd_mc=mc_file["XicSignal"].arrays()


# pd_mc.columns ## the suffix MC indicates the generated quantity


from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml import analysis_utils
from hipe4ml import plot_utils

hdl_mc = TreeHandler("/home/himanshu/Sharma/hipe4ml_Files/XicSignal.root", "XicSignal")
hdl_data = TreeHandler("/home/himanshu/Sharma/hipe4ml_Files/XicBkg.root", "XicBkg")
hdl_bkg = hdl_data.apply_preselections("fM < 2.42 or fM > 2.5", inplace=False)
print('Size of samples \nData = ',len(hdl_bkg), ', MC = ', len(hdl_mc))
print('----------------------------------')



fractionData=0.1 # this is the fraction of total data used in training 
hdl_bkg.shuffle_data_frame(frac=fractionData, inplace=True)
print('Size of samples for training \nData = ',len(hdl_bkg), ', MC = ', len(hdl_mc))

## now we remove the background from the data sample
hdl_data.apply_preselections("fM > 2.42 and fM < 2.5", inplace=True)


plot_utils.plot_distr([hdl_data], 'fM', bins=30, log=0, density=0, figsize=(5, 5), alpha=0.3, grid=False);

cols_to_be_compared = ['fM','fPt','fCpa','fCpaXY','fCt','fEta','fPhi', 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fDecayLengthXYNormalised', 
                'fErrorDecayLength' , 'fErrorDecayLengthXY', 'fChi2PCA', 'fRSecondaryVertex', 'fCandidateSelFlag',
                'fNSigTpcPi0','fNSigTpcPi2','fNSigTpcPr0','fNSigTpcPr2','fNSigTpcKa1',
                'fNSigTofPi0','fNSigTofPi2','fNSigTofPr0','fNSigTofPr2','fNSigTofKa1']


plot_utils.plot_distr([hdl_mc, hdl_bkg], cols_to_be_compared, 
                      bins=30, labels=['Signal', 'Background'],
                      log=1, density=True, figsize=(20, 20), alpha=0.3, grid=False);


plot_utils.plot_corr([hdl_mc, hdl_bkg], cols_to_be_compared, labels=['Signal', 'Background']);

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

import xgboost as xgb
n_estimators=200
max_depth=3
learning_rate=0.01

xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='error')

model_hdl = ModelHandler(xgb_model, training_cols)


import optuna
from optuna.samplers import RandomSampler

N_JOBS=-1

hyper_pars_ranges = {'n_estimators': (20, 300), 'max_depth': (1, 3), 'learning_rate': (0.01, 0.9)}
rnd_study = model_hdl.optimize_params_optuna(train_test_data, hyper_pars_ranges, cross_val_scoring='roc_auc', timeout=60, n_jobs=N_JOBS, n_trials=20, direction='maximize', optuna_sampler=RandomSampler())# tpe_time = time.time() - start
trials_rnd = rnd_study.trials
trials_array_tpe = np.array([t.values[0] for t in trials_rnd])
plt.plot(trials_array_tpe, 'o', label='TPEsampler', alpha=0.2)

model_hdl.fit(train_test_data[0], train_test_data[1])

# Create a sub-model with a specific number of boosting rounds for prediction
n_trees_for_prediction = 50  # Specify the number of trees for prediction
# sub_model = xgb_model.get_booster().slice((0, n_trees_for_prediction))

score_test = model_hdl.predict(train_test_data[2])

#### plot the score distribution
plt.hist(score_test, bins=100, label='full sample', alpha=0.5, density=True);
plt.legend();

plt.hist(score_test[train_test_data[3]==0], bins=100, label='background', alpha=0.5, density=True);
plt.hist(score_test[train_test_data[3]==1], bins=100, label='signal', alpha=0.5, density=True);
plt.legend();

plot_utils.plot_roc(train_test_data[3], score_test);

score_train = model_hdl.predict(train_test_data[0])
plot_utils.plot_roc(train_test_data[1], score_train);

plot_utils.plot_output_train_test(model_hdl, train_test_data, density=True, bins=100, logscale=True);

plot_utils.plot_feature_imp(train_test_data[2], train_test_data[3], model_hdl) 

hdl_data_pt = hdl_data.apply_preselections("fPt > 1", inplace=False)
hdl_data_pt.apply_model_handler(model_hdl)
plt.hist(hdl_data_pt.apply_preselections("model_output>0", inplace=False)["fM"], bins=50);
plt.axvline(2.467, 0, 1, label='pyplot vertical line',color='r')

hdl_data.apply_model_handler(model_hdl)
plt.hist(hdl_data.apply_preselections("model_output>0", inplace=False)["fM"], bins=50);
plt.axvline(2.467, 0, 1, label='pyplot vertical line',color='r')

plt.show()
