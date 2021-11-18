#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#export
import csv
import numpy as np  # http://www.numpy.org
import ast
from datetime import datetime
from math import log, floor, ceil
import random
import numpy as np


# Modify the Utility class's methods. You can also add additional methods as required but don't change existing methods' arguments.

# In[ ]:


#export
class Utility(object):
    
    # This method computes entropy for information gain
    def entropy(self, class_y):
        # Input:            
        #   class_y         : list of class labels (0's and 1's)

        # TODO: Compute the entropy for a list of classes
        #
        # Example:
        #    entropy([0,0,0,1,1,1,1,1,1]) = 0.918 (rounded to three decimal places)

        entropy = 0
        ### Implement your code here
        #############################################

        pass
        #############################################
        return entropy


    def partition_classes(self, X, y, split_attribute, split_val):
        # Inputs:
        #   X               : data containing all attributes
        #   y               : labels
        #   split_attribute : column index of the attribute to split on
        #   split_val       : a numerical value to divide the split_attribute

 

        # TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
        # 
        # Split_val should be a numerical value
        # For example, your split_val could be the mean of the values of split_attribute
        #
        # You can perform the partition in the following way
        # Numeric Split Attribute:
        #   Split the data X into two lists(X_left and X_right) where the first list has all
        #   the rows where the split attribute is less than or equal to the split value, and the 
        #   second list has all the rows where the split attribute is greater than the split 
        #   value. Also create two lists(y_left and y_right) with the corresponding y labels.

 

        '''
        Example:

 

        X = [[3, 10],                 y = [1,
             [1, 22],                      1,
             [2, 28],                      0,
             [5, 32],                      0,
             [4, 32]]                      1]

 

        Here, columns 0 and 1 represent numeric attributes.

 

        Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
        Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.

 

        X_left = [[3, 10],                 y_left = [1,
                  [1, 22],                           1,
                  [2, 28]]                           0]

 

        X_right = [[5, 32],                y_right = [0,
                   [4, 32]]                           1]

 

        ''' 

        X_left = []
        X_right = []

        y_left = []
        y_right = []
        ### Implement your code here
        #############################################

        pass
        #############################################
        return (X_left, X_right, y_left, y_right)


    def information_gain(self, previous_y, current_y):
        # Inputs:
        #   previous_y: the distribution of original labels (0's and 1's)
        #   current_y:  the distribution of labels after splitting based on a particular
        #               split attribute and split value

        # TODO: Compute and return the information gain from partitioning the previous_y labels
        # into the current_y labels.
        # You will need to use the entropy function above to compute information gain
        # Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

        """
        Example:

        previous_y = [0,0,0,1,1,1]
        current_y = [[0,0], [1,1,1,0]]

        info_gain = 0.45915
        """

        info_gain = 0
        ### Implement your code here
        #############################################

        pass
        #############################################
        return info_gain


    def best_split(self, X, y):
        # Inputs:
        #   X       : Data containing all attributes
        #   y       : labels
        #   TODO    : For each node find the best split criteria and return the split attribute, 
        #             spliting value along with  X_left, X_right, y_left, y_right (using partition_classes) 
        #             in the dictionary format {'split_attribute':split_attribute, 'split_val':split_val, 
        #             'X_left':X_left, 'X_right':X_right, 'y_left':y_left, 'y_right':y_right, 'info_gain':info_gain}
        '''

        Example: 

        X = [[3, 10],                 y = [1, 
             [1, 22],                      1, 
             [2, 28],                      0, 
             [5, 32],                      0, 
             [4, 32]]                      1] 

        Starting entropy: 0.971 

        Calculate information gain at splits: (In this example, we are testing all values in an 
        attribute as a potential split value, but you can experiment with different values in your implementation) 

        feature 0:  -->    split_val = 1  -->  info_gain = 0.17 
                           split_val = 2  -->  info_gain = 0.01997 
                           split_val = 3  -->  info_gain = 0.01997 
                           split_val = 4  -->  info_gain = 0.32 
                           split_val = 5  -->  info_gain = 0 
                           
                           best info_gain = 0.32, best split_val = 4 


        feature 1:  -->    split_val = 10  -->  info_gain = 0.17 
                           split_val = 22  -->  info_gain = 0.41997 
                           split_val = 28  -->  info_gain = 0.01997 
                           split_val = 32  -->  info_gain = 0 

                           best info_gain = 0.4199, best split_val = 22 

 
       best_split_feature: 1  
       best_split_val: 22  

       'X_left': [[3, 10], [1, 22]]  
       'X_right': [[2, 28],[5, 32], [4, 32]]  

       'y_left': [1, 1]  
       'y_right': [0, 0, 1] 
        '''
        
        split_attribute = 0
        split_val = 0
        X_left, X_right, y_left, y_right = [], [], [], []
        ### Implement your code here
        #############################################

        pass
        #############################################


# ### Define the classes 'DecisionTree' and 'RandomForest'

# Please modify the 'DecisionTree' and 'RandomForest' classes below

# In[ ]:


#export
class DecisionTree(object):
    def __init__(self, max_depth):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        self.max_depth = max_depth
        
    	
    def learn(self, X, y, par_node = {}, depth=0):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in Utility class to train the tree
        
        # par_node is a parameter that is useful to pass additional information to call 
        # the learn method recursively. Its not mandatory to use this parameter

        # Use the function best_split in Utility class to get the best split and 
        # data corresponding to left and right child nodes
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        ### Implement your code here
        #############################################
        
        pass
        #############################################


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        ### Implement your code here
        #############################################
        
        pass
        #############################################


# In[ ]:


#export
# This starter code does not run. You will have to add your changes and
# turn in code that runs properly.

"""
Here, 
1. X is assumed to be a matrix with n rows and d columns where n is the
number of total records and d is the number of features of each record. 
2. y is assumed to be a vector of labels of length n.
3. XX is similar to X, except that XX also contains the data label for each
record.
"""

"""
This skeleton is provided to help you implement the assignment.You must 
implement the existing functions as necessary. You may add new functions
as long as they are called from within the given classes. 

VERY IMPORTANT!
Do NOT change the signature of the given functions.
Do NOT change any part of the main function APART from the forest_size parameter.  
"""


class RandomForest(object):
    num_trees = 0
    decision_trees = []

    # the bootstrapping datasets for trees
    # bootstraps_datasets is a list of lists, where each list in bootstraps_datasets is a bootstrapped dataset.
    bootstraps_datasets = []

    # the true class labels, corresponding to records in the bootstrapping datasets
    # bootstraps_labels is a list of lists, where the 'i'th list contains the labels corresponding to records in
    # the 'i'th bootstrapped dataset.
    bootstraps_labels = []

    def __init__(self, num_trees):
        # Initialization done here
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree(max_depth=10) for i in range(num_trees)]
        self.bootstraps_datasets = []
        self.bootstraps_labels = []
        
    def _bootstrapping(self, XX, n):
        # Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        #
        # TODO: Create a sample dataset of size n by sampling with replacement
        #       from the original dataset XX.
        # Note that you would also need to record the corresponding class labels
        # for the sampled records for training purposes.

        sample = [] # sampled dataset
        labels = []  # class labels for the sampled records
        ### Implement your code here
        #############################################
            
        pass
        #############################################
        return (sample, labels)

    def bootstrapping(self, XX):
        # Initializing the bootstap datasets for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        # TODO: Train `num_trees` decision trees using the bootstraps datasets
        # and labels by calling the learn function from your DecisionTree class.
        ### Implement your code here
        #############################################
        
        pass
        #############################################

    def voting(self, X):
        y = []

        for record in X:
            # Following steps have been performed here:
            #   1. Find the set of trees that consider the record as an
            #      out-of-bag sample.
            #   2. Predict the label using each of the above found trees.
            #   3. Use majority vote to find the final label for this recod.
            votes = []
            
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)

            counts = np.bincount(votes)

            if len(counts) == 0:
                # TODO: Special case
                #  Handle the case where the record is not an out-of-bag sample
                #  for any of the trees.
                # NOTE - you can add few lines of codes above (but inside voting) to make this work
                ### Implement your code here
                #############################################
                
                pass
                #############################################
            else:
                y = np.append(y, np.argmax(counts))
                
        return y

    def user(self):
        """
        :return: string
        your GTUsername, NOT your 9-Digit GTId  
        """
        ### Implement your code here
        #############################################
        return 'gburdell3'
        #############################################


# In[ ]:


#export

# TODO: Determine the forest size according to your implementation. 
# This function will be used by the autograder to set your forest size during testing
# VERY IMPORTANT: Minimum forest_size should be 10
def get_forest_size():
    forest_size = 10
    return forest_size

# TODO: Determine random seed to set for reproducibility
# This function will be used by the autograder to set the random seed to obtain the same results you achieve locally
def get_random_seed():
    random_seed = 0
    return random_seed


# ### Do not modify the below cell
# The cell below is provided to test that your random forest classifier can be successfully built and run. Similar steps will be used to build and run your code in Gradescope. Any additional testing of functions can be done in the cells below the `%run helpers/notebook2script submission` cell, as these will not be parsed by the autograder.

# In[ ]:


def run():
    np.random.seed(get_random_seed())
    # start time 
    start = datetime.now()
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels
    numerical_cols = set([i for i in range(0, 9)])  # indices of numeric attributes (columns)

    # Loading data set
    print("reading the data")
    with open("pima-indians-diabetes.csv") as f:
        next(f, None)
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                if i in numerical_cols:
                    xline.append(ast.literal_eval(line[i]))
                else:
                    xline.append(line[i])

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])

    # Initializing a random forest.
    randomForest = RandomForest(get_forest_size())

    # printing the name
    print("__Name: " + randomForest.user()+"__")

    # Creating the bootstrapping datasets
    print("creating the bootstrap datasets")
    randomForest.bootstrapping(XX)

    # Building trees in the forest
    print("fitting the forest")
    randomForest.fitting()

    # Calculating an unbiased error estimation of the random forest
    # based on out-of-bag (OOB) error estimate.
    y_predicted = randomForest.voting(X)

    # Comparing predicted and true labels
    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))

    print("accuracy: %.4f" % accuracy)
    print("OOB estimate: %.4f" % (1 - accuracy))

    # end time
    print("Execution time: " + str(datetime.now() - start))


# In[ ]:


get_ipython().run_line_magic('run', 'helpers/notebook2script submission')


# In[ ]:


# Call the run() function to test your implementation
# Use this cell and any cells below for additional testing
run()


# In[ ]:



