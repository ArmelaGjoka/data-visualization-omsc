#!/usr/bin/env python
# coding: utf-8

# # Q3 Using Scikit-Learn

# In[11]:


#export
import numpy as np
import pandas as pd
import time
import gc
import random
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# In[12]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import tests as tests


# In[13]:


#export
class GaTech():
    # Change to your GA Tech Username
    def GTusername(self):
        gt_username = "agjoka3"
        return gt_username


# In[14]:


get_ipython().run_line_magic('run', 'helpers/verify_config.py # verify the environment setup')


# # Q3.1 Data Import and Cleansing Setup

# In[33]:


#export
class Data():
    
    # points [1]
    def dataAllocation(self,path):
        # TODO: Separate out the x_data and y_data and return each
        # args: string path for .csv file
        # return: pandas dataframe, pandas series
        # -------------------------------
        # ADD CODE HERE
        data = pd.read_csv(path)
        y_data = data['y']
        x_data = data.drop(columns=['y'])
        # ------------------------------- 
        return x_data,y_data
    
    # points [1]
    def trainSets(self,x_data,y_data):
        # TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
        # Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 614.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe, pandas series, pandas series
        # -------------------------------
        # ADD CODE HERE
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=614, shuffle=True)
        # -------------------------------
        return x_train, x_test, y_train, y_test

##################################################
##### Do not add anything below this line ########
tests.dataTest(Data)
##################################################


# # Q3.2 Linear Regression 

# In[34]:


#export
class LinearRegressionModel():
    
    # points [2]
    def linearClassifier(self,x_train, x_test, y_train):
        # TODO: Create a LinearRegression classifier and train it.
        # args: pandas dataframe, pandas dataframe, pandas series
        # return: numpy array, numpy array
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return y_predict_train, y_predict_test

    # points [1]
    def lgTrainAccuracy(self,y_train,y_predict_train):
        # TODO: Return accuracy (on the training set) using the accuracy_score method.
        # Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use any method that satisfies the requriements.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE 
       
        # -------------------------------   
        return train_accuracy
    
    # points [1]
    def lgTestAccuracy(self,y_test,y_predict_test):
        # TODO: Return accuracy (on the testing set) using the accuracy_score method.
        # Note: Round the output values greater than or equal to 0.5 to 1 and those less than 0.5 to 0. You can use any method that satisfies the requriements.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return test_accuracy
    
##################################################
##### Do not add anything below this line ########
tests.linearTest(Data,LinearRegressionModel)
##################################################


# # Q3.3 Random Forest Classifier

# In[ ]:


#export
class RFClassifier():
    
    # points [2]
    def randomForestClassifier(self,x_train,x_test, y_train):
        # TODO: Create a RandomForestClassifier and train it. Set Random state to 614.
        # args: pandas dataframe, pandas dataframe, pandas series
        # return: RandomForestClassifier object, numpy array, numpy array
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return rf_clf,y_predict_train, y_predict_test
    
    # points [1]
    def rfTrainAccuracy(self,y_train,y_predict_train):
        # TODO: Return accuracy on the training set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return train_accuracy
    
    # points [1]
    def rfTestAccuracy(self,y_test,y_predict_test):
        # TODO: Return accuracy on the test set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return test_accuracy
    
# Q3.3.1 Feature Importance
    
    # points [1]
    def rfFeatureImportance(self,rf_clf):
        # TODO: Determine the feature importance as evaluated by the Random Forest Classifier.
        # args: RandomForestClassifier object
        # return: float array
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return feature_importance
    
    # points [1]
    def sortedRFFeatureImportanceIndicies(self,rf_clf):
        # TODO: Sort them in the descending order and return the feature numbers[0 to ...].
        #       Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python.
        # args: RandomForestClassifier object
        # return: int array
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return sorted_indices
    
# Q3.3.2 Hyper-parameter Tuning

    # points [2]
    def hyperParameterTuning(self,rf_clf,x_train,y_train):
        # TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
        # args: RandomForestClassifier object, pandas dataframe, pandas series
        # return: GridSearchCV object
        # 'n_estimators': [4, 16, 256]
        # 'max_depth': [2, 8, 16]
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return gscv_rfc
    
    # points [1]
    def bestParams(self,gscv_rfc):
        # TODO: Get the best params, using .best_params_
        # args:  GridSearchCV object
        # return: parameter dict
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return best_params
    
    # points [1]
    def bestScore(self,gscv_rfc):
        # TODO: Get the best score, using .best_score_.
        # args: GridSearchCV object
        # return: float
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return best_score
    
##################################################
##### Do not add anything below this line ########
tests.RandomForestTest(Data,RFClassifier)
##################################################


# # Q3.4 Support Vector Machine

# In[ ]:


#export
class SupportVectorMachine():
    
# Q3.4.1 Pre-process

    # points [1]
    def dataPreProcess(self,x_train,x_test):
        # TODO: Pre-process the data to standardize it, otherwise the grid search will take much longer.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return scaled_x_train, scaled_x_test
    
# Q3.4.2 Classification

    # points [1]
    def SVCClassifier(self,scaled_x_train,scaled_x_test, y_train):
        # TODO: Create a SVC classifier and train it. Set gamma = 'auto'
        # args: pandas dataframe, pandas dataframe, pandas series
        # return: numpy array, numpy array
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return y_predict_train,y_predict_test
    
    # points [1]
    def SVCTrainAccuracy(self,y_train,y_predict_train):
        # TODO: Return accuracy on the training set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float 
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return train_accuracy
    
    # points [1]
    def SVCTestAccuracy(self,y_test,y_predict_test):
        # TODO: Return accuracy on the test set using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float 
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return test_accuracy
    
# Q3.4.3 Hyper-parameter Tuning
    
    # points [1]
    def SVMBestScore(self, scaled_x_train, y_train):
        # TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
        # Note: Set n_jobs = -1 and return_train_score = True and gamma = 'auto'
        # args: pandas dataframe, pandas series
        # return: GridSearchCV object, float
        # -------------------------------
        svm_parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1.0]}
        # ADD CODE HERE
        
        # -------------------------------
        
        return svm_cv, best_score
    
    # points [1]
    def SVCClassifierParam(self,svm_cv,scaled_x_train,scaled_x_test,y_train):
        # TODO: Calculate the training and test set accuracy values after hyperparameter tuning and standardization. 
        # args: GridSearchCV object, pandas dataframe, pandas dataframe, pandas series
        # return: numpy series, numpy series
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return y_predict_train,y_predict_test

    # points [1]
    def svcTrainAccuracy(self,y_train,y_predict_train):
        # TODO: Return accuracy (on the training set) using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return train_accuracy

    # points [1]
    def svcTestAccuracy(self,y_test,y_predict_test):
        # TODO: Return accuracy (on the test set) using the accuracy_score method.
        # args: pandas series, numpy array
        # return: float
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return test_accuracy
    
# Q3.4.4 Cross Validation Results

    # points [1]
    def SVMRankTestScore(self,svm_cv):
        # TODO: Return the rank test score for all hyperparameter values that you obtained in Q3.4.3. The 
        # GridSearchCV class holds a 'cv_results_' dictionary that should help you report these metrics easily.
        # args: GridSearchCV object 
        # return: int array
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return rank_test_score
    
    # points [1]
    def SVMMeanTestScore(self,svm_cv):
        # TODO: Return mean test score for all of hyperparameter values that you obtained in Q3.4.3. The 
        # GridSearchCV class holds a 'cv_results_' dictionary that should help you report these metrics easily.
        # args: GridSearchCV object
        # return: float array
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return mean_test_score

##################################################
##### Do not add anything below this line ########
tests.SupportVectorMachineTest(Data,SupportVectorMachine)
##################################################


# # Q3.5 PCA

# In[ ]:


#export
class PCAClassifier():
    
    # points [2]
    def pcaClassifier(self,x_data):
        # TODO: Perform dimensionality reduction of the data using PCA.
        #       Set parameters n_components to 8 and svd_solver to 'full'. Keep other parameters at their default value.
        # args: pandas dataframe
        # return: pca_object
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return pca
    
    # points [1]
    def pcaExplainedVarianceRatio(self, pca):
        # TODO: Return percentage of variance explained by each of the selected components
        # args: pca_object
        # return: float array
        # -------------------------------
        # ADD CODE HERE
        
        # -------------------------------
        return explained_variance_ratio
    
    # points [1]
    def pcaSingularValues(self, pca):
        # TODO: Return the singular values corresponding to each of the selected components.
        # args: pca_object
        # return: float array
        # -------------------------------
        # ADD CODE HERE
       
        # -------------------------------
        return singular_values
    
##################################################
##### Do not add anything below this line ########
tests.PCATest(Data,PCAClassifier)
##################################################


# In[ ]:


get_ipython().run_line_magic('run', 'helpers/notebook2script submission')


# In[ ]:




