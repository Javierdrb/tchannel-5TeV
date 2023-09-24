from cafea.plotter.DataGraphs import *
import pickle as pkl
from config import *

#pathmodel = '/nfs/fanae/user/juanr/cafea/analysis/tt5TeV/MVA/3j1b_500_6_minusvariablesNewMlb.pkl'
#pathmodel = '/nfs/fanae/user/andreatf/cafea/cafea/analysis/tt5TeV/models/febrary/3j1b_500_6_minusvariablesNewMlb_train.pkl'

pathmodel = 'analysis/tt5TeV/models/may/3j1b_500_4_minusvariablesNewMlb_train.pkl'

random_state = 21 # 42
vars3j1b = ['3j1b_ht','3j1b_j0pt', '3j1b_mjj', '3j1b_medianDRjj', '3j1b_mlb', '3j1b_drlb', '3j1b_druumedian', '3j1b_muu']

baseweb='/nfs/fanae/user/jriego/www/public/tt5TeV/'
outpath = baseweb + datatoday +'/' + 'MVA/'

path = 'analysis/tt5TeV/histos_forTraining/'
signal = ["TTPS"]
#bkg = ["W0JetsToLNu", "W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu"]
signal = ["TT_for_training"]
bkg = ["WJetsToLNu_forTraining"]#"W0JetsToLNu, W1JetsToLNu, W2JetsToLNu, W3JetsToLNu"#WJetsToLNu, 

datadict = {'signal': signal, 'bkg': bkg}

def CheckMaxMin(modelpath, df, vars):
  model = pkl.load(open(modelpath, 'rb'))
  pred = model.predict_proba(df[vars])[:,1]
  sig = pred[df['label']==1]
  bkg = pred[df['label']==0]
  print('Min and max for signal     = (%1.2f, %1.2f)'%(min(sig), max(sig)))
  print('Min and max for background = (%1.2f, %1.2f)'%(min(bkg), max(bkg)))
def DrawPlotsModel(modelpath, vars, lev):
    outpathlev = f'{outpath}{lev}/'
    print(outpathlev)
    
    if not os.path.exists(outpathlev): os.makedirs(outpathlev)
    ### Get the dataframe and draw some characteristic plots
    df = BuildPandasDF(path, datadict, vars, even=True)
    CheckMaxMin(modelpath, df, vars)
    DrawHistos(df, vars, savefig=f'{outpathlev}histos_{lev}.png')
    DrawBoxPlots(df, vars, savefig=f'{outpathlev}boxplots_{lev}.png')
    DrawCorrelationMatrix(df, vars, savefig=f'{outpathlev}correlation_{lev}.png')


    # Load the model
    model = pkl.load(open(modelpath, 'rb'))

    ### Draw ranking
    rank = DrawRanking(model, df, vars, savefig=f'{outpathlev}ranking_{lev}.png')
    DrawPairPlots(df, list(rank.index)[:5], savefig=f'{outpathlev}pairplots_{lev}.png')

    # Confusion matrix
    y_true = df['label'].values
    y_pred = model.predict(df[vars])
    ConfusionMatrix(y_true, y_pred, savefig=f'{outpathlev}confusion_{lev}.png')

    # Histogram of probabilities for signal and background and ROC curve
    df_train, df_test = BuildPandasDF(path, datadict, vars, even=True, train_test_split=0.85, random_state=random_state)
    train_true = df_train['label'].values
    train_pred = model.predict_proba(df_train[vars])[:,1]
    test_true = df_test['label'].values
    test_pred = model.predict_proba(df_test[vars])[:,1]

    DrawROC(train_true, train_pred, test_true, test_pred, savefig=f'{outpathlev}roc_{lev}.png')
    DrawSigBkgHisto(train_true, train_pred, test_true, test_pred, savefig=f'{outpathlev}sigbkg_{lev}.png')

if __name__ == '__main__':
  # Training plots
  DrawPlotsModel(pathmodel, vars3j1b, '3j1b')
