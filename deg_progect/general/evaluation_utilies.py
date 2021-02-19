import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import stats

from sklearn.linear_model import LinearRegression

import statistics
###############evaluate model######################
#evaluate the model using different metrics
def evaluate_model (y_predicted, y_observed, compute_mean_and_std=True):
    #if y_predicted contains NaN values then don't proceed with the evaluation and return NaNs as a result.
    if (np.isnan(np.sum(y_predicted))):
        return [(np.nan, np.nan) for i in range(4)] 
    
    y_predicted = np.squeeze(np.asarray(y_predicted))
    y_observed = np.squeeze(np.asarray(y_observed))

    print("evaluate overall:  ", "pearson:",pearsonr(y_predicted.flatten(), y_observed.flatten()))

    num_of_samples = len(y_predicted)
    metrics_results = {'R2' : np.zeros((num_of_samples,)), 'pearson' : np.zeros((num_of_samples,)), 'MSE' : np.zeros((num_of_samples,)), 'MAE' : np.zeros((num_of_samples,)), 'RMSE' : np.zeros((num_of_samples,))}
    for i in range (num_of_samples):
        y_predicted_i = y_predicted[i,:]
        metrics_results['R2'][i] = r2_score(y_observed[i,:], y_predicted_i)
        metrics_results['pearson'][i] = pearsonr(y_predicted_i, y_observed[i,:])[0]
        metrics_results['MSE'][i]= mean_squared_error(y_observed[i,:], y_predicted_i)
        metrics_results['RMSE'][i]= mean_squared_error(y_observed[i,:], y_predicted_i, squared=False)
        metrics_results['MAE'][i]= mean_absolute_error(y_observed[i,:], y_predicted_i)
    

    if (compute_mean_and_std==False):
        return [metrics_results]
    #else - compute mean and std for each metric and return the results 
    means_and_stds = {}
    for key in metrics_results:
        if (key == 'pearson' or key == 'MSE' or key == 'RMSE'):
            means_and_stds[key+'_mean'] = np.mean(metrics_results[key])
            means_and_stds[key+'_std'] = np.std(metrics_results[key])
    print(pd.DataFrame([means_and_stds]).to_string(index=False))

    print("pearson Cumulative:")
    histogram = pd.Series(metrics_results['pearson']).value_counts(bins=[-1, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], sort=False).cumsum()
    print(pd.DataFrame([histogram.values], columns=histogram.index).to_string(index=False))

    return [(means_and_stds[key+'_mean'], means_and_stds[key+'_std']) for key in ['pearson','MSE', 'RMSE']]

###############################evaluate slop test#######################################
def evaluate_slope_test (y_predicted, y_observed, compute_mean_and_std=True):
    #if y_predicted contains NaN values then don't proceed with the evaluation and return NaNs as a result. 
    if (np.isnan(np.sum(y_predicted))):
        return [(np.nan, np.nan) for i in range(4)] 
    
    y_predicted = np.squeeze(np.asarray(y_predicted))
    y_observed = np.squeeze(np.asarray(y_observed))

    num_of_samples = len(y_predicted)
    #compute the slopes and the absolute error (AE) bewteen observed slope a predicted slope
    if (y_predicted.shape[1] == 9):
        t = np.matrix([1, 2, 3, 4, 5, 6, 7, 8, 10]).T
    else:
        t = np.matrix([2, 3, 4, 5, 6, 7, 8, 10]).T
    observed_slops, predicted_slops, AE_values = np.zeros((num_of_samples,)), np.zeros((num_of_samples,)), np.zeros((num_of_samples,))
    for i in range(num_of_samples):
        mdl = LinearRegression().fit(t,y_observed[i,:])
        observed_slops[i] = mdl.coef_[0] #beta-slop
        mdl = LinearRegression().fit(t,y_predicted[i,:])
        predicted_slops[i] = mdl.coef_[0] #beta-slop
        AE_values[i] = np.abs(observed_slops[i]-predicted_slops[i]) 

    return slope_test (predicted_slops, observed_slops, compute_mean_and_std, AE_values)
    
def slope_test (predicted_slops, observed_slops, compute_mean_and_std, AE_values=None):
    predicted_slops = np.squeeze(np.asarray(predicted_slops))
    observed_slops = np.squeeze(np.asarray(observed_slops))

    if (AE_values is None):
        #if AE is None, then it wasn't computed, and therfore compute it.
        num_of_samples = len(predicted_slops)
        AE_values = np.zeros((num_of_samples,))
        for i in range(num_of_samples):
            AE_values[i] = np.abs(observed_slops[i]-predicted_slops[i])

    pearson_corr= stats.pearsonr(predicted_slops, observed_slops)
    print("slope test (pearson, p-value):", pearson_corr)

    if (compute_mean_and_std==False):
        return [AE_values]
    #else - compute mean and std for the absolute error and return the results for this and the pearson test. 
    MAE = np.mean(AE_values)
    MAE_std = np.std(AE_values)
    print("slope test (MAE, std): (", MAE, MAE_std, ")")

    return [pearson_corr, (MAE, MAE_std)]

###############################evaluate nonlinear fit subset#######################################
def evaluate_linear_or_nonlinear_subset (y_predicted, y_observed, subset='nonlinear', compute_mean_and_std=True, R2_threshold=0.7):
    print('evaluate for ', subset, ' fit subset')
    y_observed, y_predicted = drop_linear_or_nonlinear_subset (y_predicted=y_predicted, y_observed=y_observed, subset_to_retain=subset, R2_threshold=R2_threshold)

    return evaluate_model (y_predicted[:,1:], y_observed[:,1:], compute_mean_and_std) #perform the evaluation without the initial point.

###############################evaluate slope test for linear fit subset#######################################
def evaluate_slope_test_for_linear_or_nonlinear_subset (y_predicted, y_observed, subset='linear', compute_mean_and_std=True, R2_threshold=0.7):
    print('evaluate slope test for ', subset, ' fit subset')
    y_observed, y_predicted = drop_linear_or_nonlinear_subset (y_predicted=y_predicted, y_observed=y_observed, subset_to_retain=subset, R2_threshold=R2_threshold)

    return evaluate_slope_test (y_predicted, y_observed, compute_mean_and_std)

###############################drop linear or nonlinear subset#######################################
def drop_linear_or_nonlinear_subset (y_predicted, y_observed, subset_to_retain, R2_threshold):
    y_observed = np.squeeze(np.asarray(y_observed))

    num_of_samples = len(y_observed)
    #compute the slopes and the absolute error (AE) bewteen observed slope a predicted slope
    if (y_predicted.shape[1] == 9):
        t = np.matrix([1, 2, 3, 4, 5, 6, 7, 8, 10]).T
    else:
        t = np.matrix([2, 3, 4, 5, 6, 7, 8, 10]).T
    samples_indexs_to_delete_list = []
    for i in range(num_of_samples):
        mdl = LinearRegression().fit(t,y_observed[i,:])
        R2 = mdl.score(t, y_observed[i,:])
        if (subset_to_retain =='linear'):
            if (R2 < R2_threshold):
                samples_indexs_to_delete_list.append(i)
        else:
            if (R2 >= R2_threshold):
                samples_indexs_to_delete_list.append(i)
    
    y_observed = np.delete(y_observed, samples_indexs_to_delete_list, axis=0)
    y_predicted = np.delete(y_predicted, samples_indexs_to_delete_list, axis=0)

    return y_observed, y_predicted

###############################compute Linear regression slopes#######################################
def compute_LR_slopes (values_array):
    num_of_samples = len(values_array)
    #compute the slopes 
    if (values_array.shape[1] == 9):
        t = np.matrix([1, 2, 3, 4, 5, 6, 7, 8, 10]).T
    else:
        t = np.matrix([2, 3, 4, 5, 6, 7, 8, 10]).T
    slops = np.zeros((num_of_samples,))
    for i in range(num_of_samples):
        mdl = LinearRegression().fit(t,values_array[i,:])
        slops[i] = mdl.coef_[0] #beta-slope

    return slops
