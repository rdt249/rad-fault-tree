import numpy as np
import pandas as pd
pd.options.plotting.backend = 'plotly'
from copy import copy

def read_spenvis(file):
    lines = open(file).read().split('\n')
    data = []
    new = {}
    start = 0
    skips = 0
    headers = None
    for i,line in enumerate(lines):
        if skips > 0: skips -= 1; continue
        line = line.replace('\'','')
        if line.startswith('*'): start = i + int(line.split(',')[1].split(',')[0])
        elif line.startswith('SPENVIS'): new['META'] = line
        elif line.startswith('PS Annotation'): skips = int(line.split(',')[1]); headers = []
        elif i < start: 
            if headers is not None: 
                if line.split(',')[-1] in new.keys():
                    new['SPECTRUM'] = [f"{new['MOD_ABB']} {x} {new[line.split(',')[-1]][-1]}" for x in new[line.split(',')[-1]][:-1]]
                    headers += new['SPECTRUM']
                else: headers.append(f"{line.split(',')[0]} ({line.split(',')[1].strip()})")
            else:
                if line.split(',')[1].strip() == '-1': new[line.split(',')[0]] = line.split(',')[2]
                elif line.split(',')[1].strip() == '1': 
                    try: new[line.split(',')[0]] = float(line.split(',')[2])
                    except ValueError: new[line.split(',')[0]] = line.split(',')[2]
                else: new[line.split(',')[0]] = [x.strip() for x in line.split(',')[2:]]
        elif line in ['End of Block','End of File']:
            # print('Reading',file.split('/')[-1],'table',len(data),'lines',start,i,new['PLT_HDR'] if 'PLT_HDR' in new.keys() else '')
            new['DF'] = pd.read_csv(file,skiprows=start,nrows=i-start,header=None)
            try: new['DF'].columns = headers
            except ValueError:
                if 'PLT_LEG' in new.keys(): new['DF'].columns = headers[:-1] + [x.strip() for x in new['PLT_LEG']]
            headers = None
            data.append(new)
            new = {}
    return data

def get(spenvis,segment,shielding='0.370 mm'):
    orbit = read_spenvis(spenvis + '/spenvis_sao.txt')[segment]
    orbit['DF']['Latitude (deg)'] = np.around(orbit['DF']['Latitude (deg)'],2)
    orbit['DF']['SegmentNum'] = segment
    orbit['DF']['SegmentName'] = orbit['ORB_HDR']
    orbit['DF']['Time (hrs)'] = (orbit['DF']['MJD (days)'] - orbit['DF']['MJD (days)'].min()) * 24
    orbit['DF']['Time (s)'] = orbit['DF']['Time (hrs)'] * 3600
    
    niel = read_spenvis(spenvis + '/spenvis_nio.txt')

    tfluence = niel[segment]
    atten = tfluence['DF'][shielding] / tfluence['DF']['Unshielded']
    tflux = read_spenvis(spenvis + '/spenvis_spp.txt')[segment]
    tflux['DF'][tflux['SPECTRUM']] *= atten.values

    atten = niel[-4]['DF'][shielding] / niel[-4]['DF']['Unshielded']
    sfluence = read_spenvis(spenvis + '/spenvis_sef.txt')[0]
    sflux = read_spenvis(spenvis + '/spenvis_seo.txt')[segment]
    sflux['DF'] *= list(sfluence['DF']['IFlux (cm!u-2!n)'] / sfluence['DF']['Attenuation ()'] * atten / (orbit['MIS_DUR']*24*3600))

    gcr = read_spenvis(spenvis + '/spenvis_nlof_srimsi.txt')[3 * segment]
    gcr['DF'] = gcr['DF'].pivot(columns='LET (MeV cm!u2!n g!u-1!n)',values='IFlux (m!u-2!n sr!u-1!n s!u-1!n)').fillna(method='bfill').iloc[0]
    gcr['DF'] = pd.concat([gcr['DF']] * len(orbit['DF']),axis=1).T.reset_index(drop=True)
    gcr['DF'].columns = ['GCR ' + str(np.around(x / 1000,6)) + ' LET' for x in gcr['DF'].columns] # MeV-cm2/mg
    gcr['SPECTRUM'] = gcr['DF'].columns

    pstar = pd.read_csv('radstats/pstar.txt',sep=' ',skiprows=8,header=None)
    pstar.columns = ['Energy','LET',2]
    pstar = pstar.pivot(columns='Energy',values='LET').fillna(method='bfill').iloc[0]
    tlet = pstar[[float(x) for x in tflux['ENERGY'][:-1]]].values
    ttid = tflux['DF'][tflux['SPECTRUM']].cumsum().mul(tlet).sum(axis=1) * 1.6e-8 / 1000 # krad
    slet = pstar[[float(x) for x in sflux['ENERGY'][:-1]]].values
    stid = sflux['DF'][sflux['SPECTRUM']].cumsum().mul(slet).sum(axis=1) * 1.6e-8 / 1000 # krad
    glet = [float(x.split(' ')[1]) for x in gcr['SPECTRUM']] # MeV-cm2/g
    gtid = gcr['DF'][gcr['SPECTRUM']].cumsum().mul(glet).sum(axis=1) * 1.6e-8 / 1000 # krad
    tid = pd.DataFrame({'TTID':ttid,'STID':stid,'GTID':gtid,'TID':ttid + stid + gtid})
    tid[['dTTID','dSTID','dGTID','dTID']] = tid[['TTID','STID','GTID','TID']].diff(periods=1)

    scale = niel[0]['NIE_RCT']
    damage = [scale * float(x) for x in tflux['ENERGY'][:-1]]
    tddd = tflux['DF'][tflux['SPECTRUM']].cumsum().mul(damage).sum(axis=1)
    damage = [scale * float(x) for x in sflux['ENERGY'][:-1]]
    sddd = sflux['DF'][sflux['SPECTRUM']].cumsum().mul(damage).sum(axis=1)
    ddd = pd.DataFrame({'TDDD':tddd,'SDDD':sddd,'DDD':tddd + sddd})
    ddd[['dTDDD','dSDDD','dDDD']] = ddd[['TDDD','SDDD','DDD']].diff(periods=1)

    df = pd.concat([orbit['DF'],tflux['DF'],sflux['DF'],gcr['DF'],tid,ddd],axis=1)
    df.to_csv(f'trajectory/segment{segment}.csv',index=False)
    return df,{'orbit':orbit,'tflux':tflux,'sflux':sflux,'gcr':gcr,'tid':tid,'ddd':ddd}

def dwell(env,orbits=1,days=None):
    if days is not None: orbits = days / (max(env['MJD (days)']) - min(env['MJD (days)']))
    orbits, remainder = int(orbits), orbits - int(orbits)
    single = copy(env)
    for col in ['MJD (days)','Time (hrs)','Time (s)']:
        single[col] = (max(single[col]) - min(single[col])) / len(single)
    result = pd.concat([single] * orbits,ignore_index=True)
    result = pd.concat([result,single.iloc[0:int(len(single) * remainder)]],ignore_index=True)
    result['MJD (days)'] = result['MJD (days)'].cumsum() - result['MJD (days)'][0] + env['MJD (days)'][0]
    result['Time (hrs)'] = result['Time (hrs)'].cumsum() - result['Time (hrs)'][0]
    result['Time (s)'] = result['Time (s)'].cumsum() - result['Time (s)'][0]
    result[['TTID','STID','TID']] = result[['dTTID','dSTID','dTID']].cumsum().ffill()
    result[['TDDD','SDDD','DDD']] = result[['dTDDD','dSDDD','dDDD']].cumsum().ffill()
    return result

def stitch(spenvis,segments,shielding='0.370 mm'):
    # segments is a list of tuples [(segment #, segment days),(segment #, segment days) ... ]
    mission = []
    initial = 0
    for segment in segments:
        env,details = get(spenvis,segment[0],shielding)
        env = dwell(env,days=segment[1])
        if segment == 0: initial = env['MJD (days)'] = env['MJD (days)'][0]
        env[['MJD (days)','Time (hrs)','Time (s)']] = env[['MJD (days)','Time (hrs)','Time (s)']].diff(1).bfill()
        mission.append(env)
    mission = pd.concat(mission,ignore_index=True)
    mission['MJD (days)'] = mission['MJD (days)'].cumsum() + initial
    mission[['Time (hrs)','Time (s)']] = mission[['Time (hrs)','Time (s)']].cumsum()
    mission[['TTID','STID','TID']] = mission[['dTTID','dSTID','dTID']].cumsum()
    mission[['TDDD','SDDD','DDD']] = mission[['dTDDD','dSDDD','dDDD']].cumsum()
    return mission