# Supervised ML: Random Forest Regression for Dispersion Curve Prediction
"""
Created on Thurs Mar 25 11:06 2021
Author: WSH
"""

#import libraries
import pandas as pd
import numpy as np
from numpy import savetxt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Import the DMS dataset
data = pd.read_csv('DMS_Dataset_trim.csv')
data.head()

#Un-normalized data
inputs = data.drop(labels=['SV_1500', 'SV_2000', 'SV_2500', 'SV_3000', 'SV_3250', 'SV_3500', 'SV_3750', 'SV_4000', 'Compound'], axis = 'columns')
targets = data.drop(labels=['m/z', 'CCS', 'Classifier', 'Compound'], axis = 'columns')

i = 0
while i < 100:
    #split the dataset into training and test datasets
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(inputs, targets, test_size = 0.02, random_state = i)
    
    a = np.array(y_train)
    b = np.array(y_test)
    c = np.array(x_test_raw)
    
    y_train_1500 = a[:,0]
    y_train_2000 = a[:,1]
    y_train_2500 = a[:,2]
    y_train_3000 = a[:,3]
    y_train_3250 = a[:,4]
    y_train_3500 = a[:,5]
    y_train_3750 = a[:,6]
    y_train_4000 = a[:,7]
    
    y_test_1500 = b[:,0]
    y_test_2000 = b[:,1]
    y_test_2500 = b[:,2]
    y_test_3000 = b[:,3]
    y_test_3250 = b[:,4]
    y_test_3500 = b[:,5]
    y_test_3750 = b[:,6]
    y_test_4000 = b[:,7]
    
    mz = c[:,0]
    CCS = c[:,1]
    Classifier = c[:,2]
         
    x_train = x_train_raw.drop(labels=['Classifier'], axis = 'columns')
    x_test = x_test_raw.drop(labels=['Classifier'], axis = 'columns')
    
    #create a Random Forest regressor object from Random Forest Regressor class
    RFReg = RandomForestRegressor(n_estimators = 200, random_state = i) #n-estimators is the number of trees.
    
    #fit the random forest regressor with training data represented by x_train and y_train  
    RFReg.fit(x_train, y_train_1500)
    y_pred_1500 = RFReg.predict((x_test)) #predicted CV from test dataset
    
    #create a Random Forest regressor object from Random Forest Regressor class
    RFReg = RandomForestRegressor(n_estimators = 200, random_state = i) #n-estimators is the number of trees.
    
    #fit the random forest regressor with training data represented by x_train and y_train  
    RFReg.fit(x_train, y_train_2000)
    y_pred_2000 = RFReg.predict((x_test)) #predicted CV from test dataset
    
    #create a Random Forest regressor object from Random Forest Regressor class
    RFReg = RandomForestRegressor(n_estimators = 200, random_state = i) #n-estimators is the number of trees.
    
    #fit the random forest regressor with training data represented by x_train and y_train  
    RFReg.fit(x_train, y_train_2500)
    y_pred_2500 = RFReg.predict((x_test)) #predicted CV from test dataset

    #create a Random Forest regressor object from Random Forest Regressor class
    RFReg = RandomForestRegressor(n_estimators = 200, random_state = i) #n-estimators is the number of trees.
    
    RFReg.fit(x_train, y_train_3000)
    y_pred_3000 = RFReg.predict((x_test)) #predicted CV from test dataset

    #create a Random Forest regressor object from Random Forest Regressor class
    RFReg = RandomForestRegressor(n_estimators = 200, random_state = i) #n-estimators is the number of trees.
    
    RFReg.fit(x_train, y_train_3250)
    y_pred_3250 = RFReg.predict((x_test)) #predicted CV from test dataset

    #create a Random Forest regressor object from Random Forest Regressor class
    RFReg = RandomForestRegressor(n_estimators = 200, random_state = i) #n-estimators is the number of trees.
    
    RFReg.fit(x_train, y_train_3500)
    y_pred_3500 = RFReg.predict((x_test)) #predicted CV from test dataset

    #create a Random Forest regressor object from Random Forest Regressor class
    RFReg = RandomForestRegressor(n_estimators = 200, random_state = i) #n-estimators is the number of trees.
    
    RFReg.fit(x_train, y_train_3750)
    y_pred_3750 = RFReg.predict((x_test)) #predicted CV from test dataset

    #create a Random Forest regressor object from Random Forest Regressor class
    RFReg = RandomForestRegressor(n_estimators = 200, random_state = i) #n-estimators is the number of trees.
    
    RFReg.fit(x_train, y_train_4000)
    y_pred_4000 = RFReg.predict((x_test)) #predicted CV from test dataset
    
    pred_arr = [mz, CCS, Classifier, y_pred_1500, y_pred_2000, y_pred_2500, y_pred_3000, y_pred_3250, y_pred_3500, y_pred_3750, y_pred_4000]
    pred_arr_t = np.transpose(pred_arr)
    
    test_arr = [mz, CCS, Classifier, y_test_1500, y_test_2000, y_test_2500, y_test_3000, y_test_3250, y_test_3500, y_test_3750, y_test_4000]
    test_arr_t = np.transpose(test_arr)
    
    savetxt('Test_unguide_out_%i.csv' % i, test_arr_t, delimiter=',', header="m/z, CCS, Classifier, SV_1500, SV_2000, SV_2500, SV_3000, SV_3250, SV_3500, SV_3750, SV_4000")
    savetxt('Prediction_unguide_out_%i.csv' % i, pred_arr_t, delimiter=',', header="m/z, CCS, Classifier, SV_1500, SV_2000, SV_2500, SV_3000, SV_3250, SV_3500, SV_3750, SV_4000")
    
    i += 1

