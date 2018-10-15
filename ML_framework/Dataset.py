# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 09:38:49 2018

@author: reza
"""

import scipy.io as spio
import os
import pickle
import numpy as np
class Train_And_Test_Data(object):

    """
    Class for Full model LSTM decoder


    """

    def __init__(self,path='Data33.mat'):
        self.path=path
        

    def load_data(self):
        mat=spio.loadmat(self.path,squeeze_me=True)
        x_data=mat['MSTrain'][:,1:63].astype('float32')
        y_data=mat['MSTrain'][:,63:67].astype('float32')
        self.x_data=x_data
        self.y_data=y_data
        

    def min_and_max(self):
        # Calculate Min and max 
        
        self.MinX=np.min(self.y_data[:,0],axis=0)
        self.MinY=np.min(self.y_data[:,1],axis=0)
        self.MaxX=np.max(self.y_data[:,0],axis=0)
        self.MaxY=np.max(self.y_data[:,1],axis=0)
        self.VMinX=np.min(self.y_data[:,2],axis=0)
        self.VMinY=np.min(self.y_data[:,3],axis=0)
        self.VMaxX=np.max(self.y_data[:,2],axis=0)
        self.VMaxY=np.max(self.y_data[:,3],axis=0)

    def Dataset(self,PTr=0.85,PTe=0.15,PatchL=5000):
        NT=np.floor(self.x_data.shape[0]*PTe).astype('int32') ## Number Of Test
        NTr=np.floor(self.x_data.shape[0]*PTr).astype('int32')
        Patch=PatchL
        #Generate for maze classification section
        y_data2=np.zeros((self.y_data.shape))
        y_data2[:NTr,0]=np.divide(self.y_data[:NTr,0]-self.MinX,self.MaxX-self.MinX)
        y_data2[:NTr,1]=np.divide(self.y_data[:NTr,1]-self.MinY,self.MaxY-self.MinY)
        y_data2[:NTr,2]=np.divide(self.y_data[:NTr,2]-self.VMinX,self.VMaxX-self.VMinX)
        y_data2[:NTr,3]=np.divide(self.y_data[:NTr,3]-self.VMinY,self.VMaxY-self.VMinY)
        
        ArmInd=np.zeros((self.x_data.shape[0],4))
        y=self.y_data
        for i in range(self.x_data.shape[0]):
            if y[i,0] >= 165 and y[i,0] <= 200 and y[i,1] >= 60 and y[i,1] <= 165:
                ArmInd[i,0]=1
            elif y[i,0] > 200 and y[i,0] <= 240 and y[i,1] >= 60 and y[i,1] <= 165:
                ArmInd[i,1]=1
            elif y[i,0] > 240 and y[i,0] <= 265 and y[i,1] >= 60 and y[i,1] <= 165:
                ArmInd[i,2]=1
            else:
                ArmInd[i,3]=1

        self.x_data[self.x_data ==0]=-1

        ### Create training and test data
        Arm_train=np.zeros((np.int32(np.floor(NTr/Patch)),Patch,4))
        X_train=np.zeros((np.int32(np.floor(NTr/Patch)),Patch,62))
        y_train=np.zeros((np.int32(np.floor(NTr/Patch)),Patch,4))
        
        
        Arm_train=ArmInd[:NTr,:].reshape(1,NTr,4)
        X_train=self.x_data[:NTr,:].reshape(1,NTr,62)
        y_train=y_data2[:NTr,:].reshape(1,NTr,4)
        X_test=self.x_data[self.y_data.shape[0]-NT:].reshape(1,NT,62)
        Arm_test=ArmInd[self.y_data.shape[0]-NT:].reshape(1,NT,4)
        
        y_test=self.y_data[self.y_data.shape[0]-NT:].reshape(1,NT,4)
    
        print ('X_train : Train spiking data shape: ', X_train.shape)
        print ('Arm_train : Train maze labels shape: ', Arm_train.shape)
        print ('y_train : train position data shape: ', y_train.shape)
        print ('X_test : Test spiking data shape: ', X_test.shape)
        print ('Arm_test : Test maze labels shape: ', Arm_test.shape)
        print ('y_test : Test position data shape: ', y_test.shape)
        
        return [X_train,Arm_train,y_train,X_test,Arm_test,y_test]