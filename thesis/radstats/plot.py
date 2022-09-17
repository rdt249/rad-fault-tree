from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import numpy as np
import pandas as pd
pd.options.plotting.backend = 'plotly'

from radstats import models

dir = ''

layout = {
    'margin':{'l':80,'r':80,'t':50,'b':50},
    'template':'plotly_white',
    'paper_bgcolor':'rgba(0,0,0,0)',
    'plot_bgcolor':'rgba(0,0,0,0)',
    'showlegend':False
}

def rows(figs,titles=None,sharex=True):
    if titles is None: titles = [''] * len(figs)
    fig = make_subplots(rows=len(figs),shared_xaxes=sharex,subplot_titles=titles)
    for i in range(len(figs)):
        line_styles = cycle(['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'])
        #for d in figs[i].data: d.line["dash"] = next(line_styles)
        for trace in figs[i].update_traces(legendgroup=i).select_traces():
            fig.add_trace(trace,row=i+1,col=1)
        fig.update_yaxes(type=figs[i].layout.yaxis.type,exponentformat='E',row=i+1,col=1)
        fig.update_xaxes(type=figs[i].layout.xaxis.type,exponentformat='E',row=i+1,col=1)
    fig.update_layout(layout)
    fig.update_annotations(font={'size':12})
    return fig

def timeline(df):
    segment = df['SegmentNum'][0]
    titles = [
        'Altitude (km)',
        'Latitude (deg)',
        'Longitude (deg)',
        'Trapped Protons (pfu) @1,10,100 MeV',
        'Solar Protons (pfu) @1,10,100 MeV',
        'GCR (pfu) @1,10,100 MeV-cm2/mg',
        'TID (krad(Si))',
        'DDD (MeV/g)'
    ]
    fig = rows([
        df.plot('Time (hrs)','Altitude (km)'),
        df.plot('Time (hrs)','Latitude (deg)'),
        df.plot('Time (hrs)','Longitude (deg)'),
        df.plot('Time (hrs)',['TRP 1.00 MeV','TRP 10.00 MeV','TRP 100.00 MeV'],log_y=True),
        df.plot('Time (hrs)',['SEP 1.00 MeV','SEP 10.00 MeV','SEP 100.00 MeV'],log_y=True),
        df.plot('Time (hrs)',['GCR 1.0057 LET','GCR 10.088 LET','GCR 100.07 LET'],log_y=True),
        df.plot('Time (hrs)','TID'),
        df.plot('Time (hrs)','DDD')
    ],titles=titles)
    fig.write_image(dir + '/timeline.png',scale=3)
    return fig

def spectra(df):
    segment = df['SegmentNum'][0]
    titles = [
        'Trapped Protons (pfu) vs. Energy (MeV)',
        'Solar Protons (pfu) vs. Energy (MeV)',
        'GCR (pfu) vs. LET (MeV-cm2/mg)'
    ]
    cols = df.columns
    tspec,sspec,gspec = [],[],[]
    for col in cols:
        if col.startswith('TRP'): tspec.append(col)
        if col.startswith('SEP'): sspec.append(col)
        if col.startswith('GCR'): gspec.append(col)
    fig = rows([
        df.plot(x=[float(x.split(' ')[1]) for x in tspec],y=df[tspec].mean(),log_y=True,log_x=True),
        df.plot(x=[float(x.split(' ')[1]) for x in sspec],y=df[sspec].mean(),log_y=True,log_x=True),
        df.plot(x=[float(x.split(' ')[1]) for x in gspec],y=df[gspec].mean(),log_y=True,log_x=True)
    ],titles=titles,sharex=False)
    fig.write_image(dir + '/spectra.png',scale=3)
    return fig

def cross_sections(events):
    figs = []
    titles = []
    line = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
    for source in events:
        fig = go.Figure()
        if source == 'GCR':
            titles.append('Cross Section (cm2) vs. LET (MeV-cm2/mg)')
            x = np.geomspace(0.1,1e3,1000)
            f = models.weibull
            yscale = 'log'
        elif source == 'Protons':
            titles.append('Cross Section (cm2) vs. Energy (MeV)')
            x = np.geomspace(0.1,1e3,1000)
            f = models.weibull
            yscale = 'log'
        elif source == 'Dose':
            titles.append('Failure Probability vs. TID (krad)')
            x = np.geomspace(0.1,1e3,1000)
            f = models.lognormal
            yscale = 'linear'
        l = 0
        for event in events[source]:
            params = [events[source][event][key] for key in events[source][event].keys()]
            fig.add_trace(go.Scatter(x=x,y=f(x,*params),line={'dash':line[l]},name=event))
            #l += 1
        fig.update_xaxes(type='log',exponentformat='E',dtick=1)
        fig.update_yaxes(type=yscale,exponentformat='E',dtick=1)
        figs.append(fig)
    fig = rows(figs,titles=titles,sharex=False)
    fig.update_layout(layout)
    fig.write_image(dir + '/cross_sections.png',scale=3)
    return fig

def probs(elements,labels={'value':'Probability','variable':'Element'}):
    fig = px.line(elements,log_y=True,labels=labels)
    fig.update_yaxes(exponentformat='E',dtick=1)
    fig.update_layout(layout,showlegend=True)
    fig.write_image(dir + '/probability.png',scale=3)
    return fig

def heatmap(data,log=False,labels={'color':'Probability'}):
    if log:
        data = np.log(data.astype(np.float64)).replace(-np.inf,np.nan)
        data = data.fillna(data.min().min())
    fig = px.imshow(data.T,color_continuous_scale='viridis',labels=labels)
    if log:
        fig.layout.coloraxis.colorbar.tickvals = np.linspace(data.min().min(),data.max().max(),5)
        fig.layout.coloraxis.colorbar.ticktext = [f'{np.exp(x):.2e}' for x in fig.layout.coloraxis.colorbar.tickvals]
    fig.update_layout(layout)
    return fig