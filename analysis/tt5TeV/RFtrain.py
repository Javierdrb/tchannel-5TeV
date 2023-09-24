from PrepareDatasets import *
#from NN import *
import datetime
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('template')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from pylab import rcParams
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FormatStrFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
#import xgboost as xgb

### Define path, signa, bkg, variables...
path = '/nfs/fanae/user/juanr/cafea/histos5TeV/'#/nfs/fanae/user/juanr/cafea/histos/tt5TeV/forNN/'
path='/mnt_pool/c3_users/user/juanr/cafea/histos5TeV/forTraining/'
path='/mnt_pool/c3_users/user/juanr/cafea/histos5TeV/13feb2023/'
path='histos_forTraining/'
signal = "TT_for_training"
bkg = "WJetsToLNu_forTraining"#"W0JetsToLNu, W1JetsToLNu, W2JetsToLNu, W3JetsToLNu"#WJetsToLNu, 
vars_train = ['A_ht', 'A_sumAllPt', 'A_leta', 'A_j0pt', 'A_mjj', 'A_medianDRjj', 'A_drlb']
vars_train = ['A_njets','A_ht', 'A_sumAllPt', 'A_leta', 'A_j0pt', 'A_mjj', 'A_medianDRjj', 'A_drlb', 'A_minDRjj','A_j0eta','A_ptjj','A_nbtags']
vars_train = ['2j1b_ht', '2j1b_sumAllPt', '2j1b_leta', '2j1b_j0pt', '2j1b_mjj', '2j1b_medianDRjj', '2j1b_drlb', '2j1b_minDRjj','2j1b_j0eta','2j1b_ptjj','2j1b_nbtags']
vars_train = ['3j1b_ht', '3j1b_sumAllPt', '3j1b_leta', '3j1b_j0pt', '3j1b_mjj', '3j1b_medianDRjj', '3j1b_drlb', '3j1b_minDRjj','3j1b_j0eta','3j1b_ptjj']

#febrary
#vars_train = ['3j1b_ht', '3j1b_st', '3j1b_sumAllPt', '3j1b_j0pt', '3j1b_u0pt', '3j1b_ptjj', '3j1b_mjj', '3j1b_medianDRjj', '3j1b_minDRjj', '3j1b_mlb', '3j1b_drlb', '3j1b_druumedian', '3j1b_muu']
vars_train = ['3j1b_ht','3j1b_j0pt', '3j1b_mjj', '3j1b_medianDRjj', '3j1b_mlb', '3j1b_drlb', '3j1b_druumedian', '3j1b_muu']

#november (all)
#vars_train = ['3j1b_ht', '3j1b_st', '3j1b_sumAllPt', '3j1b_j0pt', '3j1b_u0pt', '3j1b_ptjj', '3j1b_mjj', '3j1b_medianDRjj', '3j1b_minDRjj', '3j1b_mlb', '3j1b_drlb', '3j1b_druumedian', '3j1b_muu']
#vars_train=['3j2b_ht', '3j2b_st', '3j2b_sumAllPt', '3j2b_j0pt', '3j2b_j0eta', '3j2b_ptjj', '3j2b_mjj', '3j2b_medianDRjj', '3j2b_minDRjj', '3j2b_mlb', '3j2b_ptsumveclb', '3j2b_drlb']

#vars_train = ['3j1b_ht', '3j1b_st', '3j1b_sumAllPt', '3j1b_j0pt', '3j1b_u0pt', '3j1b_ptjj', '3j1b_mjj', '3j1b_medianDRjj', '3j1b_minDRjj', '3j1b_ptsumveclb', '3j1b_drlb', '3j1b_druu', '3j1b_druumedian', '3j1b_muu', '3j1b_ptuu']
#vars_train=['3j2b_ht', '3j2b_st', '3j2b_sumAllPt', '3j2b_j0pt', '3j2b_j0eta', '3j2b_ptjj', '3j2b_mjj', '3j2b_medianDRjj', '3j2b_minDRjj', '3j2b_ptsumveclb', '3j2b_drlb']

#vars_train = ['3j2b_ht', '3j2b_st', '3j2b_sumAllPt', '3j2b_j0pt']# '3j2b_u0pt', '3j2b_ptjj', '3j2b_mjj', '3j2b_medianDRjj', '3j2b_minDRjj', '3j2b_mlb', '3j2b_ptsumveclb', '3j2b_drlb']

### Training parameters
trainFrac = 0.85
nest=200
depth=4
### Get the data
'''
signal = toList(signal); bkg = toList(bkg); vars_train = toList(vars_train)
datasetsSignal = pandas.concat([LoadColumns(path, s, vars_train, isSignal=True) for s in signal])
datasetsBkg    = pandas.concat([LoadColumns(path, b, vars_train, isSignal=False) for b in bkg], ignore_index=True)
X_train, X_test, y_train, y_test = train_test_split(datasetsBkg[vars_train], datasetsBkg['label'], test_size=0.3, random_state=4)
df_train_b=pd.concat([X_train,y_train], axis=1)
df_test_b=pd.concat([X_test,y_test], axis=1)
X_train, X_test, y_train, y_test = train_test_split(datasetsSignal[vars_train], datasetsSignal['label'], test_size=0.3, random_state=4)
df_test_g=pd.concat([X_test,y_test], axis=1)
df_train_g=pd.concat([X_train,y_train], axis=1)
#df_train_g=df_train_g.iloc[:int(df_train_g.shape[0]/2),:] 
df_train=pd.concat([df_train_b,df_train_g],axis=0)
df_test=pd.concat([df_test_b,df_test_g],axis=0)
'''
df_train, df_test = BuildDataset(path, signal, bkg, vars_train, trainFrac, 21, nData=None)

### Create the model
name='3j1b_%s_%s_minusvariablesNewMlb_train'%(nest,depth)

#model=DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1,class_weight='balanced')
model = RandomForestClassifier(n_estimators=nest, max_depth=depth,class_weight='balanced')
#model = GradientBoostingClassifier(n_estimators=nest, max_depth=depth)
model.n_jobs=8

### train!
print('training...')
model.fit(df_train[vars_train], df_train['label'])  
print("Running on test sample. This may take a moment.")
probs = model.predict_proba(df_test[vars_train])#predict probability over test sample
probs_train = model.predict_proba(df_train[vars_train])#predict probability over test sample
pred_y_train= model.predict(df_train[vars_train])
pred_y= model.predict(df_test[vars_train])
AUC = roc_auc_score(df_test["label"], probs[:,1])
AUC_train = roc_auc_score(df_train["label"], probs_train[:,1])

print("Test Area under Curve = {0}".format(AUC))
print("Train Area under Curve = {0}".format(AUC_train))

# ROC
fpr, tpr, _ = roc_curve(df_test["label"], probs[:,1]) #extract true positive rate and false positive rate
fpr2, tpr2, _ = roc_curve(df_train["label"], probs_train[:,1]) #extract true positive rate and false positive rate
plt.figure(figsize=(7,7))
plt.rcParams.update({'font.size': 15}) #Larger font size
plt.plot(fpr, tpr, color='crimson', lw=2, label='ROC curve test (area = {0:.4f})'.format(AUC))
plt.plot(fpr2, tpr2, color='blue', lw=1, label='ROC curve train (area = {0:.4f})'.format(AUC_train))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.title(r"$\bf{CMS}$", fontsize=20, loc='left')
plt.savefig('models/may/%s_ROCcurve_CMS.pdf'%name)
plt.savefig('models/may/%s_ROCcurve_CMS.png'%name)

# probabilities
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 15}) #Larger font size
df_test["sigprob"] = probs[:,1] #save probabilities in df
df_train["sigprob"] = probs_train[:,1] #save probabilities in df

back = np.array(df_test["sigprob"].loc[df_test["label"]==0].values)
sign = np.array(df_test["sigprob"].loc[df_test["label"]==1].values)
back2 = np.array(df_train["sigprob"].loc[df_train["label"]==0].values)
sign2 = np.array(df_train["sigprob"].loc[df_train["label"]==1].values)
plt.hist(back, 20, color='blue', edgecolor='blue', lw=2, label='Background (test)', alpha=0.3, density=True)
plt.hist(sign, 20, color='red', edgecolor='red', lw=2, label='Signal (test)', alpha=0.3, density=True)
plt.hist(back2, 20, color='green',histtype='step', lw=2, label='Background (train)', density=True)
plt.hist(sign2, 20, color='brown',histtype='step', lw=2, label='Signal (train)',  density=True)
plt.title(r"$\bf{CMS}$", fontsize=20, loc='left')
plt.xlim([0, 1])
plt.xlabel('Event probability of being classified as signal')
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig('models/may/%s_probs.pdf'%name)
plt.savefig('models/may/%s_probs.png'%name)
plt.show()

# ranking
feature_importances = pd.DataFrame(model.feature_importances_,index = df_test[vars_train].columns,columns=['importance']).sort_values('importance',ascending=False)                                                            
print(feature_importances)

# confusion matrix
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_test, pred_y))
mostrar_resultados(df_test['label'], pred_y)
plt.savefig('models/may/%s_matrix.pdf'%name)
plt.savefig('models/may/%s_matrix.png'%name)


### Save the model
#from sklearn.externals import joblib
#joblib.dump(model, 'models/may/%s.joblib'%name) 
#joblib.dump(model, 'models/may/%s.pkl'%name) 
import pickle
pickle.dump(model, open('models/may/%s_p2v2.pkl'%name, 'wb'),protocol=2) 
pickle.dump(model, open('models/may/%s.pkl'%name, 'wb')) 
print('models/may/%s_p2v2.pkl'%name)


#sig, bkg = GetSigBkgProb(model, testdl)


