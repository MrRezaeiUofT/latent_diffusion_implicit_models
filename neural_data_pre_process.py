
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append('..')
# load_folder=''
load_folder='/Users/mohammad.rezaei/Desktop/LDIDP/'
# dataset='s1'
# dataset='m1'
dataset='hc'

if dataset == 's1':
    with open(load_folder + 'example_data_s1.pickle', 'rb') as f:
        #     neural_data,vels_binned=pickle.load(f,encoding='latin1')
        neural_data, vels_binned = pickle.load(f)

if dataset == 'm1':
    with open(load_folder + 'example_data_m1.pickle', 'rb') as f:
        #     neural_data,vels_binned=pickle.load(f,encoding='latin1')
        neural_data, vels_binned = pickle.load(f)

if dataset == 'hc':
    with open(load_folder + 'example_data_hc.pickle', 'rb') as f:
        #     neural_data,pos_binned=pickle.load(f,encoding='latin1')
        neural_data, pos_binned = pickle.load(f)


#Remove neurons with too few spikes in HC dataset
if dataset=='hc':
    nd_sum=np.nansum(neural_data,axis=0)
    rmv_nrn=np.where(nd_sum<100)
    neural_data=np.delete(neural_data,rmv_nrn,1)

#The covariate is simply the matrix of firing rates for all neurons over time
X_kf=neural_data

#For the Kalman filter, we use the position, velocity, and acceleration as outputs
#Ultimately, we are only concerned with the goodness of fit of velocity (s1 or m1) or position (hc)
#But using them all as covariates helps performance

if dataset=='s1' or dataset=='m1':

    #We will now determine position
    pos_binned=np.zeros(vels_binned.shape) #Initialize
    pos_binned[0,:]=0 #Assume starting position is at [0,0]
    #Loop through time bins and determine positions based on the velocities
    for i in range(pos_binned.shape[0]-1):
        pos_binned[i+1,0]=pos_binned[i,0]+vels_binned[i,0]*.05 #Note that .05 is the length of the time bin
        pos_binned[i+1,1]=pos_binned[i,1]+vels_binned[i,1]*.05

    #We will now determine acceleration
    temp=np.diff(vels_binned,axis=0) #The acceleration is the difference in velocities across time bins
    acc_binned=np.concatenate((temp,temp[-1:,:]),axis=0) #Assume acceleration at last time point is same as 2nd to last

    #The final output covariates include position, velocity, and acceleration
    y_kf=np.concatenate((pos_binned,vels_binned,acc_binned),axis=1)


if dataset=='hc':

    temp=np.diff(pos_binned,axis=0) #Velocity is the difference in positions across time bins
    vels_binned=np.concatenate((temp,temp[-1:,:]),axis=0) #Assume velocity at last time point is same as 2nd to last

    temp2=np.diff(vels_binned,axis=0) #The acceleration is the difference in velocities across time bins
    acc_binned=np.concatenate((temp2,temp2[-1:,:]),axis=0) #Assume acceleration at last time point is same as 2nd to last

    #The final output covariates include position, velocity, and acceleration
    y_kf=np.concatenate((pos_binned,vels_binned,acc_binned),axis=1)

if dataset=='hc':
    rmv_time=np.where(np.isnan(y_kf[:,0]) | np.isnan(y_kf[:,1]))
    X_kf=np.delete(X_kf,rmv_time,0)
    y_kf=np.delete(y_kf,rmv_time,0)

if dataset=='hc':
    X_kf=X_kf[:int(.8*X_kf.shape[0]),:]
    y_kf=y_kf[:int(.8*y_kf.shape[0]),:]



valid_range_all=[[0,.1],[.1,.2],[.2,.3],[.3,.4],[.4,.5],
                 [.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1]]
testing_range_all=[[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],
                 [.6,.7],[.7,.8],[.8,.9],[.9,1],[0,.1]]
#Note that the training set is not aways contiguous. For example, in the second fold, the training set has 0-10% and 30-100%.
#In that example, we enter of list of lists: [[0,.1],[.3,1]]
training_range_all=[[[.2,1]],[[0,.1],[.3,1]],[[0,.2],[.4,1]],[[0,.3],[.5,1]],[[0,.4],[.6,1]],
                   [[0,.5],[.7,1]],[[0,.6],[.8,1]],[[0,.7],[.9,1]],[[0,.8]],[[.1,.9]]]
num_folds=len(valid_range_all) #Number of cross validation folds

y_kf_test_all=[]
y_kf_train_all=[]
y_kf_valid_all=[]


num_examples = X_kf.shape[0]  # number of examples (rows in the X matrix)

for i in range(1):  # Loop through the folds

    ######### SPLIT DATA INTO TRAINING/TESTING/VALIDATION #########

    # Note that all sets have a buffer of 1 bin at the beginning and 1 bin at the end
    # This makes it so that the different sets don't include overlapping neural data

    # This differs from having buffers of "num_bins_before" and "num_bins_after" in the other datasets,
    # which creates a slight offset in time indexes between these results and those from the other decoders

    # Get testing set for this fold
    testing_range = testing_range_all[i]
    testing_set = np.arange(int(np.round(testing_range[0] * num_examples)) + 1,
                            int(np.round(testing_range[1] * num_examples))  - 1)

    # Get validation set for this fold
    valid_range = valid_range_all[i]
    valid_set = np.arange(int(np.round(valid_range[0] * num_examples)) + 1,
                          int(np.round(valid_range[1] * num_examples)) - 1)

    # Get training set for this fold
    # Note this needs to take into account a non-contiguous training set (see section 3B)
    training_ranges = training_range_all[i]
    for j in range(len(training_ranges)):  # Go through different separated portions of the training set
        training_range = training_ranges[j]
        if j == 0:  # If it's the first portion of the training set, make it the training set
            training_set = np.arange(int(np.round(training_range[0] * num_examples)) + 1,
                                     int(np.round(training_range[1] * num_examples))  - 1)
        if j == 1:  # If it's the second portion of the training set, concatentate it to the first
            training_set_temp = np.arange(int(np.round(training_range[0] * num_examples))  + 1,
                                          int(np.round(training_range[1] * num_examples)) - 1)
            training_set = np.concatenate((training_set, training_set_temp), axis=0)

    # Get training data
    X_kf_train = X_kf[training_set, :]
    y_kf_train = y_kf[training_set, :]

    # Get validation data
    X_kf_valid = X_kf[valid_set, :]
    y_kf_valid = y_kf[valid_set, :]

    # Get testing data
    X_kf_test = X_kf[testing_set, :]
    y_kf_test = y_kf[testing_set, :]

    ##### PREPROCESS DATA #####

    # Z-score "X_kf" inputs.
    X_kf_train_mean = np.nanmean(X_kf_train, axis=0)  # Mean of training data
    X_kf_train_std = np.nanstd(X_kf_train, axis=0)  # Stdev of training data
    X_kf_train = (X_kf_train - X_kf_train_mean) / X_kf_train_std  # Z-score training data
    X_kf_test = (
                            X_kf_test - X_kf_train_mean) / X_kf_train_std  # Preprocess testing data in same manner as training data
    X_kf_valid = (
                             X_kf_valid - X_kf_train_mean) / X_kf_train_std  # Preprocess validation data in same manner as training data

    # Zero-center outputs
    y_kf_train_mean = np.nanmean(y_kf_train, axis=0)  # Mean of training data outputs
    y_kf_train = y_kf_train - y_kf_train_mean  # Zero-center training output
    y_kf_test = y_kf_test - y_kf_train_mean  # Preprocess testing data in same manner as training data
    y_kf_valid = y_kf_valid - y_kf_train_mean  # Preprocess validation data in same manner as training data


''' save dataset'''
dataset_dic= {'X_train': X_kf_train,
             'X_test': X_kf_test,
             'X_val':X_kf_valid,
             'Y_train':y_kf_train,
             'Y_test':y_kf_test,
             'Y_val':y_kf_valid}


if dataset == 's1':
    pickle.dump(dataset_dic, open(load_folder+"example_data_s1_pp.p", "wb"))
    dataset_v2 = pickle.load(open(load_folder+"example_data_s1_pp.p", "rb"))

if dataset == 'm1':

    pickle.dump(dataset_dic, open(load_folder+"example_data_m1_pp.p", "wb"))
    dataset_v2 = pickle.load(open(load_folder+"example_data_m1_pp.p", "rb"))

if dataset == 'hc':

    pickle.dump(dataset_dic, open(load_folder+"example_data_hc_pp.p", "wb"))
    dataset_v2 = pickle.load(open(load_folder+"example_data_hc_pp.p", "rb"))

