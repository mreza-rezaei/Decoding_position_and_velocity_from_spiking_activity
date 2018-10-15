# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:11:34 2018

@author: reza
"""


import numpy as np

########## R-squared (R2) ##########

def get_R2(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s

######### Abs Distance ################

def get_AbsDis(y_test,y_test_pred):
    Dis_list=[] #Initialize a list that will contain the Dis for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute DIs for each output
        Dis=np.mean(np.abs((y_test_pred[:,i]-y_test[:,i])))
        Dis_list.append(Dis) #Append Dis of this output to the list
    Dis_array=np.array(Dis_list)
    return Dis_array #Return an array of RMSEs
	

########## RMSE #####################

def get_RMSE(y_test,y_test_pred):
    RMSE_list=[] #Initialize a list that will contain the RMSEs for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute RMSE for each output
        RMSE=np.sqrt(np.mean((y_test_pred[:,i]-y_test[:,i])**2))
        RMSE_list.append(RMSE) #Append RMSE of this output to the list
    RMSE_array=np.array(RMSE_list)
    return RMSE_array #Return an array of RMSEs    

