# before importing from flask, import and model data
import numpy as np
import pandas as pd
# import dill as pickle
from bricdatascience.models import moddate, babynamepop, knnestimate
# from sklearn.neighbors import NearestNeighbors
from bricdatascience import app
from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure, output_file, save, show, ColumnDataSource
from bokeh.models import HoverTool, NumeralTickFormatter
from bokeh.embed import components

### flask page functions
@app.route('/')
def main():
    return redirect('/index')

### landing page
@app.route('/index')
def index():
    # modification date
    updated = moddate()
    return render_template('index.html', updated=updated)
 
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
        
    
