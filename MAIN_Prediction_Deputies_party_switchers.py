# Required Libraries are imported
import warnings  
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import confusion_matrix
import time
import itertools
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from typing import Tuple
import copy as cp
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
import mrmr
from mrmr import mrmr_classif
from datetime import datetime, timedelta
from sklearn.utils import shuffle
from sklearn.feature_selection import mutual_info_classif
import random


from auxilary_function_compute_features_voting_deputies_prediction import compute_features_voting_deputies_prediction




def cross_val_function(model, X : np.array, y : np.array):

    kfold = KFold(n_splits=5, shuffle=True)

    model_ = cp.deepcopy(model)
    
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    
    
    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
        actual_classes = np.append(actual_classes, test_y)

        clf_N = model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        y_pred = clf_N.predict(test_X)
        
        if("conf_mat" in locals()):
            conf_mat = conf_mat + confusion_matrix(test_y, y_pred)
        else:
            conf_mat = confusion_matrix(test_y, y_pred)

    return actual_classes, predicted_classes, conf_mat






#%% DATA LOADING - we do not own the data therefore we can't put them in Github. 
# They are freely available at https://dati.camera.it/it/

# load df_deputies which is a dataframe that contains the following columns: 
# 'persona' -> link to the Deputy's personal page on the Chamber of Deputies website
# 'cognome' -> Deputy surname
# 'nome' -> Deputy name
# 'info' -> Deputy CV
# 'dataNascita' -> Deputy date of birth
# 'luogoNascita' -> Deputy place of birth
# 'genere' -> Deput gender
# 'inizioMandato' -> Deput date of beginning of service
# 'fineMandato' -> Deput date of service end
# 'collegio' -> Deput region of election
# 'lista' -> Deput party of election
# 'numeroMandati' -> Deput number of served terms
# 'Gruppi' -> Deput parties during their service
# 'Ngruppi' -> Deput number parties during their service
           

###############################################################################
# load df_votes which is a dataframe that contains the following columns: 
# 'persona' -> link to the Deputy's personal page on the Chamber of Deputies website
# 'cognome' -> Deputy surname
# 'nome' -> Deputy name
# a set of columns named as the date of each voting sessions in the Chamber of 
# Deputies. Each entry of these columns contains a list that, for each Deputy, 
# specify: 
# 1) number of presence to voting sessions
# 2) number of absence to voting sessions
# 3) number of votes in agreement with party majority
# 4) number of abstensions
# 5) number of votes in favour of a law that was then approved by the Chamber 
# 6) number of votes in opposition to a law
# 7) number of votes in favour to a law
# 8) number of votes casted in seret ballots
#%%

moving_wind=60

df_parlamentari_single_group = df_parlamentari[df_parlamentari.Ngruppi <= 1] # select the MPs who did NOT switch party

# randomly sample 5 times a subset of the single group MPs (which are the largest group) 
random_indexes_single_group=[]
for trials in range(5):
    random_indexes_single_group.append(random.sample(range(0, len(df_parlamentari_single_group)), len(df_parlamentari_single_group)))





# in this loop we compute the prediction of ............................................

chance_level =[]
accuracy_on_test =[]
accuracy_std_on_test =[]
confusion_matrix =[]

for offset_window in tqdm(range(0,420,3)):
    
    print('processing %d of 116' %(offset_window/15)) 
    
    chance_level_trials=[]
    accuracy_on_test_trials=[]
    accuracy_std_on_test_trials=[]
    confusion_matrix_trials=[]
    
    for trials in range(5): # repeat the classificaiton considering a two months windows with an offset of offset_window to the party switch
        moving_wind=60 # two months windows in which extracting the voting features of Deputies
        [single_group_MPs, pluri_group_MPs] = compute_features_voting_deputies_prediction(df_parlamentari,df_votes,moving_wind,offset_window,random_indexes_single_group[trials])

    
        X = single_group_MPs.append(pluri_group_MPs) # data of voting and non-voting features of Deputies    
        labels=np.concatenate((np.zeros(len(single_group_MPs)),np.ones(len(pluri_group_MPs))),axis=0) # labels
     
        chosen_idx_test = np.random.choice(len(single_group_MPs), replace = False, size = int(np.round(len(single_group_MPs)/3))) # randomly select test set
        X_test = single_group_MPs.iloc[chosen_idx_test] # data test set
        df_testdata = X_test.append(pluri_group_MPs.iloc[chosen_idx_test]) # data test set
        df_testlabel = np.concatenate((np.zeros(len(chosen_idx_test)),np.ones(len(chosen_idx_test))),axis=0) # label test set
        
        chosen_idx_train = np.asarray(list(set(np.arange(0,len(single_group_MPs),1)) - set(chosen_idx_test))) # randomly select training set
        X_train = single_group_MPs.iloc[chosen_idx_train] # data training set
        df_traindata = X_train.append(pluri_group_MPs.iloc[chosen_idx_train]) # data training set
        df_trainlabel = np.concatenate((np.zeros(len(chosen_idx_train)),np.ones(len(chosen_idx_train))),axis=0) # labels training set


        new_dummy_classifier = DummyClassifier(strategy="stratified") # build dummy classifier for computing the chance level
        new_dummy_classifier.fit(df_traindata, df_trainlabel) 
        chance_level_trials.append(new_dummy_classifier.score(df_testdata, df_testlabel)) # chance level
        
        classifiers = RandomForestClassifier() # build the random forest classifier
        classifiers = classifiers.fit(df_traindata, df_trainlabel) # fit the random forest classifier to training data
        y_pred = classifiers.predict(df_testdata) # access the performance of the classifier on test data
        accuracy_on_test_trials.append(accuracy_score(y_pred, df_testlabel)) # compute the accuracy on test set
        accuracy_std_on_test_trials.append(cross_val_score(classifiers,X,labels, cv=5).std()) # compute the standard deviation on the 5-fold cross validation of the accuracy on test set

        # this function return the confusion matrix of the classifer accessed with a 5-fold cross validation
        actual_classes, predicted_classes, confusion_forest = cross_val_function(RandomForestClassifier(), KFold(n_splits=5, shuffle=True), X.to_numpy(), labels) 
        
        confusion_matrix_trials.append(confusion_forest)
        
    chance_level.append(np.mean(chance_level_trials))
    accuracy_on_test.append(np.mean(accuracy_on_test_trials))
    accuracy_std_on_test.append(np.mean(accuracy_std_on_test_trials))
    confusion_matrix.append(np.mean(confusion_matrix_trials,axis=0))
    
#%% PLOTTING RESULTS


plt.figure()
x=np.arange(0,len(accuracy_on_test),3)
plt.plot(x,accuracy_on_test)
plt.fill_between(x, [x-y for x,y in zip(accuracy_on_test,accuracy_std_on_test)], [x+y for x,y in zip(accuracy_on_test,accuracy_std_on_test)],alpha=0.4)
plt.axhline(y = 0.5, color = 'r', linestyle = '--')
plt.plot(x,accuracy_on_test)
plt.ylim(0.4,1)

###############################################################################

acc_conf_mat=[]
for mat in range(len(confusion_matrix)):
    acc_conf_mat.append((confusion_matrix[mat][0,0]+confusion_matrix[mat][1,1])/(sum(sum(confusion_matrix[mat]))))

plt.figure()
plt.plot(x,acc_conf_mat)
plt.axhline(y = 0.5, color = 'r', linestyle = '--')
plt.ylim(0.4,1)





