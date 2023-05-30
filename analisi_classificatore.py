# required libraries 
import warnings
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV, RFE, VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.feature_selection import mutual_info_classif
from sklearn.dummy import DummyClassifier
from datetime import datetime, timedelta
from sklearn.utils import shuffle
from sklearn.feature_selection import mutual_info_classif
from compute_features_voting_deputies import compute_features_voting_deputies

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


def analysis_MP_already_out(X,y_already_out):

    # let's compute the chance level
    dummy_clf = DummyClassifier()
    dummy_clf.fit(X, y_already_out)
    chance_level_already_out = dummy_clf.score(X, y_already_out)

    warnings.filterwarnings('ignore')

    classifiers = RandomForestClassifier()
        
    from sklearn.model_selection import train_test_split
    X_train_already_out, X_test_already_out, y_train_already_out, y_test_already_out = train_test_split(X, y_already_out, test_size = 0.30, stratify=y_already_out)
    
    accuracy_tmp=[]
    accuracy_error_tmp=[]
    for trials in range(10):
        clf = classifiers.fit(X_train_already_out, y_train_already_out)
        y_pred = clf.predict(X_test_already_out)
        scores = accuracy_score(y_pred, y_test_already_out)
        scores_std = cross_val_score(clf,X_test_already_out,y_test_already_out, cv=5).std()
        
        accuracy_tmp.append(scores[0])
        accuracy_error_tmp.append(scores_std[0])

    return chance_level_already_out,np.mean(accuracy_tmp),np.mean(accuracy_error_tmp)


#%% DATA LOADING

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


chance_level_already_out=[]
accuracy_test_already_out=[]
acc_test_std_already_out=[]
selected_features_list_already_out=[]
confusion_matrix=[]

for alpha_mesi in tqdm(range(121)):
    print('processing %d of 121' %(alpha_mesi+1) 
    
    mesi_thr = datetime(2013, 4, 9)+timedelta(days=15*(alpha_mesi)) # consider the voting until mesi_thr
    mesi_thr=mesi_thr.date()

    has_MP_left_party, single_party_MP, pluri_party_MP = copmute_features_voting_deputies(df_deputies,df_votes,mesi_thr)

    X = single_party_MP.append(pluri_party_MP) # voting and non voting features of both MP groups
    
    y_already_out = np.zeros(len(pluri_party_MP)) # set to 1 the label of those MP that have already left the parlamentary group
    y_already_out = [1 if x else 0 for x in has_MP_left_party]
    y_already_out=np.concatenate((np.zeros(len(single_party_MP)),y_already_out),axis=0) 
    
    selected_features_list_already_out.append(mrmr_classif(X=X, y=y_already_out, K=20)) # mrmr classifier for feature importance
    
    try: 
        chance_level,accuracy_test_set,accuracy_test_set_std = analysis_MP_already_out(X,y_already_out)
    
        chance_level_already_out.append(chance_level)
        accuracy_test_already_out.append(accuracy_test_set)
        acc_test_std_already_out.append(accuracy_test_set_std)
    
        try:
            actual_classes, predicted_classes, confusion_forest = cross_val_function(RandomForestClassifier(), X.to_numpy(), y_already_out)
            confusion_matrix.append(confusion_forest)
        except:
            empty=np.nan
            confusion_matrix.append(empty)
    except:
        empty=np.nan
        chance_level_already_out.append(empty)
        accuracy_test_already_out.append(empty)
        acc_test_std_already_out.append(empty)
        confusion_matrix.append(empty)

###############################################################################
plt.figure()
x=np.arange(0,len(accuracy_test_already_out)*15,15)
plt.plot(x/30,accuracy_test_already_out)
plt.fill_between(x/30, [x-y for x,y in zip(accuracy_test_already_out,acc_test_std_already_out)], [x+y for x,y in zip(accuracy_test_already_out,acc_test_std_already_out)],alpha=0.4)
plt.plot(x/30,chance_level_already_out, linestyle = '--')
plt.xlabel('months')
plt.ylabel('accuracy')
plt.show()
plt.ylim(0.65,1.05)
plt.show()
###############################################################################
plt.figure()
x=np.arange(0,len(accuracy_test_already_out)*15,15)
acc_aboce_chance = [x-y for x,y in zip(accuracy_test_already_out,chance_level_already_out)]
plt.plot(x/30,acc_aboce_chance)
plt.fill_between(x/30, [x-y for x,y in zip(acc_aboce_chance,acc_test_std_already_out)], [x+y for x,y in zip(acc_aboce_chance,acc_test_std_already_out)],alpha=0.4)
plt.plot(x/30,[0*x for x in chance_level_already_out], linestyle = '--')
plt.xlabel('months')
plt.ylabel('accuracy above chance level')
plt.show()
plt.ylim(-0.05,0.25)
plt.show()



for loop_ind in range(0,121,30): 
    try: 
        cm=(confusion_mat_randforest[loop_ind]+confusion_mat_randforest[loop_ind+1])/2
    except: 
        cm=confusion_mat_randforest[loop_ind]

    plt.figure()
    seaborn.heatmap((cm.T / np.sum(cm,1)).T,vmin=0, vmax=1)
    plt.title('confusion matrix at %d months' %(loop_ind*15/30))
    plt.text(0.25, 0.5, "%f"%cm[0,0], bbox=dict(facecolor='red', alpha=0.5))
    plt.text(1.25, 0.5, "%f"%cm[0,1], bbox=dict(facecolor='red', alpha=0.5))
    plt.text(0.25, 1.5, "%f"%cm[1,0], bbox=dict(facecolor='red', alpha=0.5))
    plt.text(1.25, 1.5, "%f"%cm[1,1], bbox=dict(facecolor='red', alpha=0.5))
    plt.show()
    
    
    
    
###############################################################################
important_features = np.zeros((14,121))
for time_points in range(121):
    if(len(selected_features_list_already_out[time_points])>0):
        important_features[int(selected_features_list_already_out[time_points][0]),righe] = 3 # most important feature
        important_features[int(selected_features_list_already_out[time_points][1]),righe] = 2 # second most important feature
        important_features[int(selected_features_list_already_out[time_points][2]),righe] = 1 # third most important feature

plt.figure()
plt.imshow(important_features,cmap='viridis')
plt.show()
plt.colorbar()
plt.title('Feature importance (first three features)')
plt.show()
