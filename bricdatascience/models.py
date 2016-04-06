'''
Functions for serving webpages.
'''
import numpy as np
import pandas as pd
import os, datetime, operator
from sklearn.neighbors import NearestNeighbors

# load data
df = pd.read_csv('EarningsTuition.csv', sep=',', encoding='utf-8')
dfraw = pd.read_csv('NationalNames.csv', sep=',', encoding='utf-8')
dfall = pd.read_csv('gda20160315.tsv', sep='\t')

# define functions
def moddate():
    t = os.path.getmtime('bricdatascience/views.py')
    modt=datetime.date.fromtimestamp(t)
    return 'Last updated: '+modt.strftime('%B %e, %Y')
  
def babynamepop(gender, popularity, earliest, latest, viewsize=20, mincount=1000):
    '''
    Operates on dataframe 'dfraw'
    Data comes from a 44mb CSV stored locally
    '''
    dfrecent = dfraw[(dfraw.Year>=int(earliest)) & (dfraw.Year<=int(latest))]
    
    dfgroup = dfrecent[['Name','Gender','Count']].groupby(['Name','Gender'], 
                        sort=False, as_index=False).sum()
    dfreccom = dfgroup[dfgroup.Count>mincount]
    if gender == 'M' or gender == 'F':
        df = dfreccom[dfreccom.Gender.values==gender]
    else:
        df = dfreccom
    poptarget = df.Count.quantile(popularity)
    dfquant = df.iloc[(df.Count-poptarget).abs().argsort()[:viewsize]]
    dfout = dfquant.sort_values(by='Count',ascending=True, 
                                inplace=False).reset_index(drop=True)
    return dfout

def knnestimate(earn, tuition, nn=5):
    '''
    Function used in earncost to retrieve colleges that most closely match the user selected alumni earnings and annual tutition.
    `df` must have column order: institution, state, tuition, 50% earnings, 75% earnings, 90% earnings.
    '''
    cols = df.columns
    dfs = {}
    xtest = np.array([tuition, earn]).reshape(1,-1)
    knn = NearestNeighbors()
    for percent, col in zip(['50%', '25%', '10%'], cols[3:]):
        X = df[['PublicPrivate', col]]
        knn.fit(X)
        _ , idxs = knn.kneighbors(xtest, nn)
        outcols = list(cols[:3])
        outcols.append(col)
        dfres = df[outcols].iloc[idxs.flatten()].copy()
        dfres.columns = ['Institution', 'State', 'Tuition ($)', 'Earnings ($)']
        dfres.sort_values('Tuition ($)', inplace=True, ascending=True)
        with pd.option_context('max_colwidth', -1):
            testhtml = dfres.to_html(index=False, escape=False, 
            classes='table table-condensed table-striped table-bordered')
        testhtml = testhtml.replace('border="1" ', '').replace('class="dataframe ', 'class="')
        testhtml = testhtml.replace(' style="text-align: right;"', '').replace('&', '&amp;')
        dfs[percent] = testhtml
    return dfs
    
def gda():
    catclr = {'Kidneys and urinary system': '#7f7f7f', 'Food, nutrition, and metabolism': '#e377c2', 'Digestive system': '#d62728', 'Not Available': '#eeeeee', 'Brain and nervous system': '#2ca02c', 'Mouth and teeth': '#17becf', 'Reproductive system': '#ffbb78', 'Lungs and breathing': '#bcbd22', 'Bones, muscles, and connective tissues': '#ff7f0e', 'Blood/lymphatic system': '#1f77b4', 'Eyes and vision': '#8c564b', 'Ear, nose, and throat': '#9467bd', 'Skin, hair, and nails': '#98df8a'}
    disease_dict = {'immune': 'Immune system', 'mouth': 'Mouth and teeth', 
                    'kidney': 'Kidneys and urinary system', 
                    'reproductive': 'Reproductive system', 
                    'metabolism': 'Food, nutrition, and metabolism', 
                    'digest': 'Digestive system', 'blood': 'Blood/lymphatic system', 
                    'bone': 'Bones, muscles, and connective tissues', 
                    'ent': 'Ear, nose, and throat', 'not': "Not Available",
                    'endocrine': 'Endocrine system (hormones)', 'all': 'All', 
                    'eye': 'Eyes and vision', 'cancer': 'Cancers', 'skin': 
                    'Skin, hair, and nails', 'heart': 'Heart and circulation', 
                    'mental': 'Mental health and behavior', 
                    'brain': 'Brain and nervous system', 'lung': 'Lungs and breathing'}
    atype_dict = {'all': 'All', 'genetic': 'GeneticVariation', 
                  'altered': 'AlteredExpression'}
    perc_dict = {'00': 'Not Selected', '01': 'Top 1%', '05': 'Top 5%', 
                 '10': 'Top 10%', '15': 'Top 15%', '20': 'Top 20%', '25': 'Top 25%'}
                 
    # create pull-down menu list
    catdict = {k: v for k, v in disease_dict.items() 
               if (v in dfall.category.unique()) or (k in 'all') or (k in 'not')}
    catlist = sorted(catdict.items(), key=operator.itemgetter(1))
    atypedict = {k: v for k, v in atype_dict.items() 
               if (v in dfall.associationType.unique()) or (k in 'all') or (k in 'not')}
    atypelist = sorted(atypedict.items(), key=operator.itemgetter(1))
    perclist = sorted(perc_dict.items(), key=operator.itemgetter(0))
    return dfall, catlist, atypelist, perclist, catclr, disease_dict, atype_dict