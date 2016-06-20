# before importing from flask, import and model data
import numpy as np
import pandas as pd
# import dill as pickle
from bricdatascience.models import moddate, babynamepop, knnestimate, gda, saformat, genidx
# from sklearn.neighbors import NearestNeighbors
from bricdatascience import app
from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure, output_file, save, show, ColumnDataSource
from bokeh.models import HoverTool, NumeralTickFormatter, TapTool, OpenURL
from bokeh.embed import components

### flask page functions
@app.route('/')
def main():
    return redirect('/index')

### landing page
@app.route('/index')
def index():
    arr = genidx()
    # modification date
    updated = moddate()
    return render_template('index.html', updated=updated, arr=arr)
 
### college rankings
@app.route('/rankings')
def rankings():
    # modification date
    updated = moddate()
    return render_template('rankings.html', updated=updated)

### loan analysis
@app.route('/loans')
def loans():
    return app.send_static_file('loan-analysis.html')
    
### expedia recommendations
@app.route('/expedia')
def expedia():
    return app.send_static_file('expedia.html')

### college earnings and cost   
@app.route('/earncost', methods=['GET', 'POST'])
def earncost(): 
    # load data
    # data = pd.read_csv(rel+'EarningsTuition.csv', sep=',')
    # get vs post
    if request.method=='GET':
        earntarget = 60000
        tuitiontarget = 10000
        nn = 5
    else:
        earntarget = int(request.form['earnings'])
        tuitiontarget = int(request.form['tuition'])
        nn = int(request.form['viewsize'])
    # run function to get tables
    dfs = knnestimate(earntarget, tuitiontarget, nn)
    # modification date
    updated = moddate()
    return render_template('earncost.html', updated=updated, dfs=dfs,
                            earnings=earntarget, tuition=tuitiontarget, viewsize=nn)


### us baby names
@app.route('/babynamespopularity', methods=['GET', 'POST'])
def babynamespopularity(): 

    # define simple function to colorcode gender
    def colors(gender):
        if gender=='M':
            clr = "#57B768"
        elif gender=='F':
            clr = "#A979BE"
        return clr

    # define simple function to position names over graph
    def calcoffset(right, mx):
        if right < 0.14 * (dfout.Count.max()-mincount):
            txtclr = "black"
            offset = max(right*1.5, right+(0.04*mx))
            align = "left"
        else:
            txtclr = "white"
            offset = min(right*0.9, right-(0.1*mx))
            align = "right"
        return pd.Series([txtclr, offset, align])

    # values for populating drop down menus
    glist = (('B','Both'), ('F','Female'), ('M','Male'))
    poplist = (('1.00','Most popular'), ('0.90', 'Top 10%'), ('0.75', 'Top 25%'), 
            ('0.50','Middle 50%'), ('0.25','Bottom 25%'), ('0.10','Bottom 10%'), 
            ('0.00','Least popular'))
    # load data
    # dfraw = pd.read_csv(rel+'NationalNames.csv', sep=',')
    # get variables
    if request.method=='GET':
        earliest=1980
        latest=2014
        mincount=10000
        viewsize=10
        gender='B'
        popularity=0
    else:
        gender = request.form['gender']
        popularity = float(request.form['popularity'])
        mincount = int(request.form['mincount'])
        viewsize = int(request.form['viewsize'])
        earliest = request.form['earliest']
        latest = request.form['latest']
    # process raw data
    dfout = babynamepop(gender, popularity, earliest=earliest, 
                    latest=latest, mincount=mincount, viewsize=viewsize)
    # add necessary columns
    dfout['Color']=dfout.Gender.apply(colors)
    dfout['Right']=dfout.Count-mincount
    mx = dfout.Count.max()-mincount
    dfout = pd.concat([dfout,dfout['Right'].apply(lambda x: calcoffset(x, mx))],axis=1)
    # rename columns with string names
    col = dfout.columns.tolist()
    col[-3:] = ['TextColor', 'Offset', 'Align']
    dfout.columns = col
    # generate plot
    output_file("babynamespopularity.html")
    source = ColumnDataSource(dfout)
    hover = HoverTool(tooltips=[("Name", "@Name"), ("Total Count", "@Count")], 
                        names=['bars'])
    # save  value for x range max value
    rt = dfout.Count.max()-mincount+1
    p = figure(width=750, height=600, y_range=(0,len(dfout)+1), 
                x_range=(0,rt), tools=[hover, 'reset','save'])
    # plot bars
    p.quad(left=[0]*viewsize, bottom=[x+0.6 for x in range(0,viewsize)], 
            top=[x+1.4 for x in range(0,viewsize)], right=dfout.Right, color=dfout.Color,
            source=source, name='bars')
    # add name labels and other formatting
    p.text(x=dfout.Offset, y=dfout.index+0.8, text=dfout.Name.tolist(), 
            text_color=dfout.TextColor.tolist(), text_align="center", 
            text_font_size="0.8em", text_font="helvetica neue")
    p.xaxis.axis_label = "Number of Babies Above Minimum Count"
    p.ygrid.grid_line_color = None
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_label_text_color = None
    p.xaxis[0].formatter = NumeralTickFormatter(format='0,0')
    # make graph responsive
    p.responsive = True
    # set plot components
    script, div = components(p)
    # modification date
    updated = moddate()

    return render_template('babynamespopularity.html', script=script, div=div, updated=updated, mincount=mincount, viewsize=viewsize, glist=glist, poplist=poplist, gcheck=gender, earliest=earliest, latest=latest, pcheck='{:.2f}'.format(popularity))
    

### genetic links to health
@app.route('/genediseaselink', methods=['GET', 'POST'])
def genediseaselink(): 
    dfall, catlist, atypelist, perclist, catclr, disease_dict, atype_dict = gda()
    dfstash = dfall.copy()
    # return user-selected data
    jump = ''
    selcat = ''
    selatype = ''
    errmsg = ''
    selperc = '00'
    if request.method=='POST':
        jump = '<script> window.location.hash="gdaplot"; </script>'
        selcat = request.form['selectioncat']
        selatype = request.form['selectiontype']
        selperc = request.form['selectioncir']
        cirrad = np.percentile(dfall['ideal'],int(selperc))
        if selcat != 'all':
            dfall = dfall[dfall['category']==disease_dict[selcat]]
        if selatype != 'all':
            dfall = dfall[dfall['associationType']==atype_dict[selatype]]
#         if selperc != '00':
        if len(dfall)==0:
            dfall = dfstash.copy()
            selcat='all'
            selatype='all'
            errmsg='No data found for selected filters'
    yy = dfall['count_total']
    xx = dfall['score_total']

    # Generate plot
    output_file("templates/index.html")
    dfall['color'] = dfall['category'].map(lambda x: catclr[x])
    source = ColumnDataSource(
        data=dict(
            x=xx,
            y=yy,
            genes=dfall['geneCount'],
            desc=dfall['diseaseName'],
            desc2=dfall['geneSymbol'],
            cat=dfall['category'],
            assoc=dfall['associationType'],
            prev=dfall['Prevalence']
        )
    )
    hover = HoverTool(tooltips=[("Disease", "@desc"), ("Category", "@cat"), 
    ("Prevalence", "@prev"), ("Gene", "@desc2"), ("Type", "@assoc"), 
    ("Association", "@x{0.000}"), ("Specificity", "@y{0.000}")], names=['pts'])

    p = figure(plot_width=900, plot_height=600, tools=['box_zoom','pan','reset',
    'save',hover,'tap'], x_range=[0,0.8], y_range=[0,0.8], 
    title='Associations above high-quality threshold')
    p.title_text_font = 'Source Sans Pro'
    p.xaxis.axis_label = 'Association Score'
    p.yaxis.axis_label = 'Specificity Score'
    p.xaxis.axis_label_text_font = 'Source Sans Pro'
    p.yaxis.axis_label_text_font = 'Source Sans Pro'
    if selperc != '00':
        xcir = np.linspace(-0.6,1,100)
        ycir = 1-np.sqrt(np.square(cirrad)-np.square(xcir-1))
        xcir = np.append(xcir, 1.0)
        ycir = np.append(ycir, 1.0)
        p.patch(xcir, ycir, line_color='#33ff33', fill_alpha=0.15, color='#ccffcc', 
        line_width=1)
    p.circle('x', 'y', size=15+dfall['PrevalenceCode']*5, fill_alpha=0.5, 
    color=dfall['color'], source=source, line_width=0.75, line_color='#000000', 
    name='pts')  
    url = "http://www.ncbi.nlm.nih.gov/pubmed/?term=@desc+@desc2"
    taptool = p.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    p.responsive = True
    script, div = components(p)
    # modification date
    updated = moddate()
    return render_template('gda.html', script=script, div=div, catlist=catlist, atypelist=atypelist, selcat=selcat, selatype=selatype, perclist=perclist, selperc=selperc, updated=updated, jumpscript=jump, errmsg = errmsg)        

    
### shelter animal outcomes
@app.route('/shelteranimals', methods=['GET', 'POST'])
def shelteranimals():
    # options
    names = ['Named', 'Unnamed']
    types = ['Dog', 'Cat']
    genders = ['Intact Female', 'Intact Male', 'Neutered Male', 'Spayed Female', 
                'Unknown']
    breeds = ['Mix', 'Pure breed', 'Pit bull']
    ageunits = ['days', 'weeks', 'months', 'years']
    # for get method
    selname = 'Named'
    seltype = 'Dog'
    selgender = 'Neutered Male'
    selbreed = 'Mix'
    selagenum = '1'
    selageunit = 'years'
    seldate = '2016-01-02'
    selhour = '17'
    selmin = '0'
    res = ''
    alert = '<div class="alert"><p class="error">&nbsp;</p></div>'
    probs = saformat(selname, seltype, selgender, selbreed, 
            selagenum+' '+selageunit, seldate, selhour, selmin)
    if request.method=='POST':
        selname = request.form['named']
        seltype = request.form['type']
        selgender = request.form['gender']
        selbreed = request.form['breed']
        selagenum = request.form['age']
        selageunit = request.form['ageunit']
        seldate = request.form['date']
        selhour = str(int(request.form['hour'])) # handle zero padding
        selmin = str(int(request.form['minute'])) # handle zero padding
        # error handling
        if seltype=='Cat' and selbreed=='Pit bull':
            probs = ['0%']*5
            alert = '<div class="alert alert-danger fade"><p class="error">Cats cannot be pit bulls! Please select again.</p></div>'
        elif any([selhour==i for i in ['1','2','3','4']]):
            selhour = '5'
            probs = saformat(selname, seltype, selgender, selbreed, 
            selagenum+' '+selageunit, seldate, selhour, selmin)
            alert = '<div class="alert alert-warning fade"><p class="error">No animals delivered at that time. Hour set to 5am.</p></div>'
        else:
            alert = '<div class="alert alert-success fade"><p class="error">Prediction successful.</p></div>'
            probs = saformat(selname, seltype, selgender, selbreed, 
            selagenum+' '+selageunit, seldate, selhour, selmin)
    # modification date
    updated = moddate()
    return render_template('shelteranimals.html', probs=probs, updated=updated, 
    names=names, selname=selname, types=types, seltype=seltype, 
    breeds=breeds, selbreed=selbreed, genders=genders, selgender=selgender, 
    selagenum=selagenum, ageunits=ageunits, selageunit=selageunit, 
    selhour=selhour, selmin=selmin, seldate=seldate, res=res, alert=alert)