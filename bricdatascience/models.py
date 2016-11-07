'''
Functions for serving webpages.
'''
import numpy as np
import pandas as pd
import os, datetime, operator, joblib, re, requests
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier

from bokeh.plotting import figure, save, show, ColumnDataSource
from bokeh.palettes import Set1
from bokeh.models import Legend
from collections import OrderedDict

# load data
df = pd.read_csv('EarningsTuition.csv', sep=',', encoding='utf-8')
dfraw = pd.read_csv('babynames.csv', sep=',')
dfall = pd.read_csv('gda20160315.tsv', sep='\t')
# gbc = joblib.load('gbc.pkl')
gbc = joblib.load('gbc_sklearn_018.pkl')

# generate index page
def genidx():
    # ordered list of projects and properties (name, pgtitle, text, img)
    arr = [
            {'name':'kingcohouseprices', 
            'title':'Predicting Home Prices', 'img':'img/house.jpg',
            'text': 'Evaluating and optimizing several machine learning models to accurately predict home prices in 2014-2015 King County, WA.',
            'doc': 'Jupyter notebook'
            },
            {'name':'genediseaselink', 
            'title':'Genetic Links to Health', 'img':'img/gda.jpg',
            'text': 'Identifying the most likely genetic cause of a disease can help pharmaceutical companies better target their R&D efforts. This data-centric approach to drug development may help bring down consumer drug prices.',
            'doc': 'Web application'
            },
            {'name':'shelteranimals',
             'title':'Shelter Animal Outcomes', 'img':'img/pet.jpg',
             'text':'Using machine learning to predict the outcome for animals dropped off at a municipal animal shelter. Inspired by a 2016 kaggle competition.',
            'doc': 'Web application'
             },
             {'name':'candlestick',
             'title':'Equity Price Movement', 'img':'img/stocks.jpg',
             'text':'Provides elementary information including prices, movement, dividends, splits, and calculated moving averages for 3000 US equities.',
             'doc': 'Web application'
             },
             {'name':'expedia',
             'title':'Travel Recommendations', 'img':'img/travel.jpg',
             'text':'Recommending hotels for Expedia users from search parameters. Inspired by a 2016 kaggle competition.',
             'doc': 'Jupyter notebook'
             },
             {'name':'babynamespopularity',
             'title':'Trends in U.S. Baby Names', 'img':'img/child.jpg',
             'text':"For anyone curious about the changing trends in name popularity for the last 100+ years.",
             'doc': 'Web application'
             },
             {'name':'earncost',
             'title':'College Outcomes & Rankings', 'img':'img/education.jpg',
             'text':'Data from the US Dept. of Education and three independent international organizations is combined to evaluate the performance of US universities.',
             'doc': 'Web application'
             },
             {'name':'loans',
             'title':'Personal Loan Outcomes', 'img':'img/money.jpg',
             'text':'A brief exploratory analysis of personal loan data, in search of insights that might aid in predicting when and what loans will default.',
             'doc': 'Jupyter notebook'
             }
             ]
    # specify striping
    for idx,item in enumerate(arr):
        if idx%2 == 0:
            item['cont'] = 'content-section-b'
        else:
            item['cont'] = 'content-section-a'
    return arr

# last modified date
def moddate():
    t = os.path.getmtime('bricdatascience/views.py')
    modt=datetime.date.fromtimestamp(t)
    return 'Last updated: '+modt.strftime('%B %e, %Y')

### baby names function
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

### college function
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

### gene-disease association
def gda():
    catclr = {'Kidneys and urinary system': '#7f7f7f', 
              'Food, nutrition, and metabolism': '#e377c2', 
              'Digestive system': '#d62728', 'Not Available': '#eeeeee', 
              'Brain and nervous system': '#2ca02c', 'Mouth and teeth': '#17becf', 
              'Reproductive system': '#ffbb78', 'Lungs and breathing': '#bcbd22', 
              'Bones, muscles, and connective tissues': '#ff7f0e', 
              'Blood/lymphatic system': '#1f77b4', 'Eyes and vision': '#8c564b', 
              'Ear, nose, and throat': '#9467bd', 'Skin, hair, and nails': '#98df8a'}
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
    
### shelter animal outcomes prediction
def predprob(df):
    # output array has alphabetical order for class names
    nparr = gbc.predict_proba(df)
    return ['{:.1%}'.format(float(p)) for p in nparr]
    
# format shelter animal info
def convage(age):
    # accepts string of format "## days/weeks/months/years"
    dperiod = {'month': 30, 'year': 365, 'years': 365, 'day': 1, 'days': 1, 'weeks': 7, 'week': 7, 'months': 30}
    num = int(re.match('\d+',age).group())
    period = re.search('\w+$', age).group()
    return int(num*dperiod[period])

def saformat(name,type,sex,breed,age,date,hour,minute):
    cols = ['Name_0', 'Name_1', 'AnimalType_Cat', 'AnimalType_Dog',
       'SexuponOutcome_Intact Female', 'SexuponOutcome_Intact Male',
       'SexuponOutcome_Neutered Male', 'SexuponOutcome_Spayed Female',
       'SexuponOutcome_Unknown', 'Breed_0', 'Breed_1', 'Breed_2', 'Year_2013',
       'Year_2014', 'Year_2015', 'Year_2016', 'Month_1', 'Month_2', 'Month_3',
       'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9',
       'Month_10', 'Month_11', 'Month_12', 'DayofMonth_1', 'DayofMonth_2',
       'DayofMonth_3', 'DayofMonth_4', 'DayofMonth_5', 'DayofMonth_6',
       'DayofMonth_7', 'DayofMonth_8', 'DayofMonth_9', 'DayofMonth_10',
       'DayofMonth_11', 'DayofMonth_12', 'DayofMonth_13', 'DayofMonth_14',
       'DayofMonth_15', 'DayofMonth_16', 'DayofMonth_17', 'DayofMonth_18',
       'DayofMonth_19', 'DayofMonth_20', 'DayofMonth_21', 'DayofMonth_22',
       'DayofMonth_23', 'DayofMonth_24', 'DayofMonth_25', 'DayofMonth_26',
       'DayofMonth_27', 'DayofMonth_28', 'DayofMonth_29', 'DayofMonth_30',
       'DayofMonth_31', 'DayofWeek_0', 'DayofWeek_1', 'DayofWeek_2',
       'DayofWeek_3', 'DayofWeek_4', 'DayofWeek_5', 'DayofWeek_6', 'Hour_0',
       'Hour_5', 'Hour_6', 'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11',
       'Hour_12', 'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17',
       'Hour_18', 'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23',
       'Minute_0', 'Minute_1', 'AgeDays']
    dfsa = pd.DataFrame(np.zeros((1,len(cols)), dtype=int), columns=cols)
    # get col name bases (before _)
    p = re.compile('\w+(?=_)')
    chops = []
    for col in cols[:-1]:
        curr = p.search(col).group()
        if curr not in chops:
            chops.append(curr)
    chops = [s+'_' for s in chops]
    chops.append(cols[-1])
    # dictionaries and datetime for setting values
    dname = {'Named':'1', 'Unnamed':'0'}
    dbreed = {'Mix':'0', 'Pure breed':'1', 'Pit bull':'2'}
    dmin = {'0':'0'}
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    # suffixes for column names
    locs = dict(zip(chops,[dname[name], type, sex, dbreed[breed], str(dt.year), 
                           str(dt.month), str(dt.day), str(dt.weekday()), hour, 
                           dmin.get(minute,'1'), '']))
    #set values
    for chop in chops:
        if chop=='AgeDays':
            dfsa.set_value(0, chop+locs[chop], convage(age))
        else:
            dfsa.set_value(0, chop+locs[chop], 1)
    nparr = gbc.predict_proba(dfsa)[0]
    return ['{:.1%}'.format(float(p)) for p in nparr]


###--------------------------------------------------------------
def df_from_ticker(ticker):
    '''
    Given a ticker, retrieves historical daily stock price data from quandl.
    '''
    # keep apikey hidden from github
    with open('apikey.txt', 'r') as f:
        apikey = f.read().strip()
    baseurl = 'https://www.quandl.com/api/v3/datasets/WIKI/'
    suffixurl = '.json?api_key='+apikey
    call = baseurl+ticker+suffixurl
    rj = requests.get(call).json()
    dataset = rj.get('dataset','error')

    #error handling (incorrect ticker)
    if dataset=='error':
        return pd.DataFrame([]), '', ''

    df = pd.DataFrame(dataset['data'], columns=dataset['column_names'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', drop=True, inplace=True)
    df.sort_index(inplace=True, ascending=True)

    df['Volume'] = df['Volume'].astype(int)
    df['Adj. Volume'] = df['Adj. Volume'].astype(int)

    name = dataset['name']
    desc = dataset['description']

    return df, name, desc

def candleplot(df):
    '''
    Produces bokeh interactive plot from equity prices.
    '''
    dfma = df['Adj. Close']
    for window in [10,20,50,200]:
        ser = pd.Series(df['Adj. Close'].rolling(window=window).mean(), name=str(window)+' day moving avg')
        dfma = pd.concat((dfma, ser), axis=1)

    # calculations
    mids = (df['Adj. Open'] + df['Adj. Close'])/2
    spans = abs(df['Adj. Close']-df['Adj. Open'])

    inc = df['Adj. Close'] > df['Adj. Open']
    dec = df['Adj. Open'] > df['Adj. Close']
    w = 16*60*60*1000 # 2/3 day in ms

    p = figure(x_axis_label='Date', x_axis_type='datetime', y_axis_label='Price ($)', 
               tools=['pan,box_zoom,reset,save,crosshair, wheel_zoom'], toolbar_location='right', 
               plot_width=950, plot_height=600)

    p.segment(df.index, df['Adj. High'], df.index, df['Adj. Low'], color="black")
    p.rect(df.index[inc], mids[inc], w, spans[inc], fill_color="#5cd65c", line_color="black")
    p.rect(df.index[dec], mids[dec], w, spans[dec], fill_color="#F2583E", line_color="black")

    lines = []
    i=0
    for col,clr in zip(dfma.columns[1:],reversed(Set1[1+len(dfma.columns[1:])])):
        lines.append(p.line(dfma.index, dfma[col], color=clr, line_width=1.5))

    legend = Legend(items=list(OrderedDict(zip(dfma.columns[1:],[[i] for i in lines])).items()), 
                    location=(55, 0), spacing=10)

    p.add_layout(legend, 'above')
    p.legend.orientation = 'horizontal'

    return p
