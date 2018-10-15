# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:11:34 2018

@author: reza
"""

import keras
from keras.layers import *
from keras.models import *
import keras.backend as K
import timeit
import numpy as np
#################### LONG SHORT TERM MEMORY (LSTM) DECODER ##########################
def Build_model(X_train,y_train,Topology): 
    
        if Topology == 1:
            # Create Position  and Velocity estimator model
            #input
            In=Input(shape=(X_train.shape[1:]),name='Input')
            # position
            P=TimeDistributed(Dense(2,activation='sigmoid'),name='P')(In)
            #velocity
            V=TimeDistributed(Dense(2,activation='sigmoid'),name='V')(In)
            #build model
            Model2=Model(inputs=In,outputs=[P,V])
            #compile model
            Model2.compile(loss={'P':'mse','V':'mse'},optimizer='rmsprop')
        elif Topology==2:
            # Create Position and Velocity estimator model
            #input
            In=Input(shape=(X_train.shape[1:]),name='Input')
            #Maze
            Mazee=LSTM(4,return_sequences=True,name='MazeSeq')(In)
            Maze=TimeDistributed(Dense(4,activation='sigmoid'),name='Maze')(Mazee)
            M=concatenate([Mazee,In],name='ConcatFeature')
            #position
            Pe=LSTM(2,return_sequences=True,name='P_Features')(M)
            P=TimeDistributed(Dense(2,activation='sigmoid'),name='P')(Pe)
            #velocity
            Ve=LSTM(2,return_sequences=True,name='V_Features')(M)
            V=TimeDistributed(Dense(2,activation='sigmoid'),name='V')(Ve)
            #build model
            Model2=Model(inputs=In,outputs=[P,V,Maze])
            #compile
            Model2.compile(loss={'P':'mse','V':'mse','Maze':'binary_crossentropy'},optimizer='rmsprop',metrics={'Maze':'accuracy'})
        elif Topology==3:
            # Create Position and Velocity estimator model
            #input
            In=Input(shape=(X_train.shape[1:]),name='Input')
            #maze
            Mazee=LSTM(4,return_sequences=True,name='MazeSeq')(In)
            Maze=TimeDistributed(Dense(2,activation='softmax'),name='Maze')(Mazee)
            #velocity
            Ve=LSTM(2,return_sequences=True,name='VelocitySeq')(In)
            V=TimeDistributed(Dense(2,activation='sigmoid'),name='Velocity')(Ve)
            #position
            M=concatenate([Mazee,In,Ve],name='ConcatFeature')
            Pe=LSTM(2,return_sequences=True,name='P_Features')(M)
            P=TimeDistributed(Dense(2,activation='sigmoid'),name='P')(Pe)
            #build model
            Model2=Model(inputs=In,outputs=[P,V,Maze])
            #compile
            Model2.compile(loss={'Velocity':'mse','P':'mse','Maze':'binary_crossentropy'},optimizer='rmsprop',metrics={'Maze':'accuracy'})
        elif Topology==4:
            # Create Position and Velocity estimator model
            #input
            In=Input(shape=(X_train.shape[1:]),name='Input')
            #maze
            Mazee=LSTM(4,return_sequences=True,name='MazeSeq')(In)
            Maze=TimeDistributed(Dense(2,activation='softmax'),name='Maze')(Mazee)
            #Accelerate
            Ae=LSTM(2,return_sequences=True,name='AcceSeq')(In)
            A=TimeDistributed(Dense(2,activation='sigmoid'),name='ACC')(Ae)
            #velocity
            M0=concatenate([A,In],name='ConcatFeature0')
            Ve=LSTM(2,return_sequences=True,name='VelocitySeq')(M0)
            V=TimeDistributed(Dense(2,activation='sigmoid'),name='Velocity')(Ve)
            #position
            M=concatenate([Maze,In,Ve],name='ConcatFeature')
            Pe=LSTM(2,return_sequences=True,name='P_Features')(M)
            P=TimeDistributed(Dense(2,activation='sigmoid'),name='P')(Pe)
            #build model
            Model2=Model(inputs=In,outputs=[P,V,A,Maze])
            #compile
            Model2.compile(loss={'Velocity':'mse','P':'mse','ACC':'mse','Maze':'binary_crossentropy'},optimizer='rmsprop',metrics={'Maze':'accuracy'})

        elif Topology==5:
            # Create Position and Velocity estimator model
            #input
            In=Input(shape=(X_train.shape[1:]),name='Input')
            #maze
            Mazee=LSTM(4,return_sequences=True,name='MazeSeq')(In)
            Maze=TimeDistributed(Dense(2,activation='softmax'),name='Maze')(Mazee)
            #Accelerate
            Ae=LSTM(2,return_sequences=True,name='AcceSeq')(In)
            A=TimeDistributed(Dense(2,activation='sigmoid'),name='ACC')(Ae)
            #velocity
            M0=concatenate([A,In],name='ConcatFeature0')
            Ve=LSTM(2,return_sequences=True,name='VelocitySeq')(M0)
            V=TimeDistributed(Dense(2,activation='sigmoid'),name='Velocity')(Ve)
            #position
            M=concatenate([Maze,In,Ve],name='ConcatFeature')
            Pe=LSTM(2,return_sequences=True,name='P_Features')(M)
            P=TimeDistributed(Dense(2,activation='sigmoid'),name='P')(Pe)
            #build model
            Model2=Model(inputs=In,outputs=[P,V,A,Maze])
            #compile
            Model2.compile(loss={'Velocity':self.penalized_loss(A),'P':self.penalized_loss(V),'ACC':'mse','Maze':'binary_crossentropy'},optimizer='rmsprop',metrics={'Maze':'accuracy'})

        else:
            print("No valid Topology")
        Model2.summary()
        return Model2

class Full_Model_LSTMDecoder(object):

    """
    Class for Full model LSTM decoder


    """

    def __init__(self,verbose=0,epochs=100,Topology=1,path='PositionEstimatorFull.h5'):
        
        
        self.verbose=verbose
        self.path=path
        self.epochs=epochs
        self.topology=Topology
    def penalized_loss(self,Pen):
        # Define cost function for bouth positions
        Pen1=(K.abs(Pen-0.5)+.1)*2
        def loss(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true)*Pen1 , axis=-1)
        return loss
    
    def create_model(self,X_train,y_train):
        
        self.model=Build_model(X_train,y_train,self.topology)
    def fit(self,X_train,y_train,Arm_train,save=False,use_pretrained=False):
        ## Fit the model    
        if self.topology == 1:
            if use_pretrained == True:
                self.model.load_weights(self.path)
            Hist=self.model.fit(X_train,[y_train[:,:,:2],y_train[:,:,2:4]],shuffle=False,epochs=self.epochs,verbose=self.verbose,batch_size=1)
            if save == True:
                self.model.save_weights(self.path)
            return Hist
        elif self.topology == 2:
            if use_pretrained == True:
                self.model.load_weights(self.path)
            Hist=self.model.fit(X_train,[y_train[:,:,:2],y_train[:,:,2:4],Arm_train],shuffle=False,epochs=self.epochs,verbose=self.verbose,batch_size=1)
            if save == True:
                self.model.save_weights(self.path)
            return Hist
        elif self.topology == 3:
            if use_pretrained == True:
                self.model.load_weights(self.path)
            Hist=self.model.fit(X_train,[y_train[:,:,:2],y_train[:,:,2:4],Arm_train],shuffle=False,epochs=self.epochs,verbose=self.verbose,batch_size=1)
            if save == True:
                self.model.save_weights(self.path)
            return Hist
        elif self.topology == 4:
            if use_pretrained == True:
                self.model.load_weights(self.path)
            Hist=self.model.fit(X_train,[y_train[:,:,:2],y_train[:,:,2:4],y_train[:,:,4:],Arm_train],shuffle=False,epochs=self.epochs,verbose=self.verbose,batch_size=1)
            if save == True:
                self.model.save_weights(self.path)
            return Hist
        elif self.topology == 5:
            if use_pretrained == True:
                self.model.load_weights(self.path)
            Hist=self.model.fit(X_train,[y_train[:,:,:2],y_train[:,:,2:4],y_train[:,:,4:],Arm_train],shuffle=False,epochs=self.epochs,verbose=self.verbose,batch_size=1)
            if save == True:
                self.model.save_weights(self.path)
            return Hist
    def save_or_load(self,save=False,use_pretrained=False):
        if use_pretrained == True:
            self.model.load_weights(self.path)
        if save == True:
            self.model.save_weights(self.path)
            
    def predict(self,X_test,y_test):
        
        self.test_model=Build_model(X_train,y_train,self.topology)
        
        self.test_model.set_weights(self.model.get_weights()) 
        if Topology ==1:
            y_predict=np.zeros((X_test.shape[1],4))
            [y_valid_predicted_lstm,VS]=Model2.predict(X_test)
            y_predict[:,:2]=y_valid_predicted_lstm[0,:,:]
            y_predict[:,2:4]=Vs[0,:,:]
            stop = timeit.default_timer()
            print('test time=%f'% (stop - start) )
        elif Topology ==2:
            y_predict=np.zeros((X_test.shape[1],8))
            [y_valid_predicted_lstm,VS,Maze]=Model2.predict(X_test)
            y_predict[:,:2]=y_valid_predicted_lstm[0,:,:]
            y_predict[:,2:4]=Vs[0,:,:]
            y_predict[:,4:]=Maze[0,:,:]
            stop = timeit.default_timer()
            print('test time=%f'% (stop - start) )
        elif Topology ==3:
            y_predict=np.zeros((X_test.shape[1],10))
            [y_valid_predicted_lstm,VS,Maze]=Model2.predict(X_test)
            y_predict[:,:2]=y_valid_predicted_lstm[0,:,:]
            y_predict[:,2:4]=Vs[0,:,:]
            y_predict[:,6:]=Maze[0,:,:]
            stop = timeit.default_timer()
            print('test time=%f'% (stop - start) )
        elif Topology ==4:
            y_predict=np.zeros((X_test.shape[1],10))
            [y_valid_predicted_lstm,VS,AS,Maze]=Model2.predict(X_test)
            y_predict[:,:2]=y_valid_predicted_lstm[0,:,:]
            y_predict[:,2:4]=Vs[0,:,:]
            y_predict[:,4:6]=As[0,:,:]
            y_predict[:,6:]=Maze[0,:,:]
            stop = timeit.default_timer()
        elif Topology ==5:
            y_predict=np.zeros((X_test.shape[1],10))
            [y_valid_predicted_lstm,VS,AS,Maze]=Model2.predict(X_test)
            y_predict[:,:2]=y_valid_predicted_lstm[0,:,:]
            y_predict[:,2:4]=Vs[0,:,:]
            y_predict[:,4:6]=As[0,:,:]
            y_predict[:,6:]=Maze[0,:,:]
            stop = timeit.default_timer()
        return [y_predict,Maze]
