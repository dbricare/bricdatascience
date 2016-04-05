'''
Functions for serving webpages.
'''
import numpy as np
import pandas as pd
import os, datetime
from sklearn.neighbors import NearestNeighbors

# load data
df = pd.read_csv('EarningsTuition.csv', sep=',', encoding='utf-8')
dfraw = pd.read_csv('NationalNames.csv', sep=',', encoding='utf-8')

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