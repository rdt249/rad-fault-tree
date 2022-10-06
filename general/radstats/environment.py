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
    orbit['DF']['Latitude (deg)'] = np.around(orbit['DF']['Latitude (deg)'],2).replace(-0,0)
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

    gflux = read_spenvis(spenvis + '/spenvis_nlof_srimsi.txt')[3 * segment]
    gflux['DF'] = gflux['DF'].pivot(columns='LET (MeV cm!u2!n g!u-1!n)',values='IFlux (m!u-2!n sr!u-1!n s!u-1!n)').fillna(method='bfill').iloc[0]
    gflux['DF'] = pd.concat([gflux['DF']] * len(orbit['DF']),axis=1).T.reset_index(drop=True)
    gflux['DF'].columns = ['GCR ' + str(np.around(x / 1000,6)) + ' LET' for x in gflux['DF'].columns] # MeV-cm2/mg
    gflux['SPECTRUM'] = gflux['DF'].columns

    pstar = pd.read_csv('radstats/pstar.txt',sep=' ',skiprows=8,header=None)
    pstar.columns = ['Energy','LET',2]
    pstar = pstar.pivot(columns='Energy',values='LET').fillna(method='bfill').iloc[0]
    tlet = pstar[[float(x) for x in tflux['ENERGY'][:-1]]].values / 1000    # MeV-cm2/mg
    trate = np.trapz(tflux['DF'][tflux['SPECTRUM']].mul(tlet).replace(np.nan,0),tlet[::-1]) * 1.6e-8   # krad/s
    tdose = orbit['DF']['Time (s)'].diff().bfill() * trate           # krad
    slet = pstar[[float(x) for x in sflux['ENERGY'][:-1]]].values / 1000    # MeV-cm2/mg
    srate = np.trapz(sflux['DF'][sflux['SPECTRUM']].mul(slet).replace(np.nan,0),slet[::-1]) * 1.6e-8    # krad/s
    sdose = orbit['DF']['Time (s)'].diff().bfill() * srate           # krad
    glet = [float(x.split(' ')[1]) for x in gflux['SPECTRUM']]              # MeV-cm2/mg
    grate = np.trapz(gflux['DF'][gflux['SPECTRUM']].mul(glet).replace(np.nan,0),glet) * 1.6e-8   # krad/s
    gdose = orbit['DF']['Time (s)'].diff().bfill() * grate           # krad
    tid = pd.DataFrame({'TRate (krad/s)':trate,'SRate (krad/s)':srate,'GRate (krad/s)':grate,'Rate (krad/s)':trate + srate + grate,
        'TDose (krad)':tdose,'SDose (krad)':sdose,'GDose (krad)':gdose,'Dose (krad)':tdose + sdose + gdose,})
    tid[['TTID (krad)','STID (krad)','GTID (krad)','TID (krad)']] = tid[['TDose (krad)','SDose (krad)','GDose (krad)','Dose (krad)']].cumsum()

    scale = niel[0]['NIE_RCT']
    damage = [scale * float(x) for x in tflux['ENERGY'][:-1]]
    trate = tflux['DF'][tflux['SPECTRUM']].mul(damage).sum(axis=1)
    tdose = orbit['DF']['Time (s)'].diff().bfill() * trate.values
    damage = [scale * float(x) for x in sflux['ENERGY'][:-1]]
    srate = sflux['DF'][sflux['SPECTRUM']].mul(damage).sum(axis=1)
    sdose = orbit['DF']['Time (s)'].diff().bfill() * srate.values
    ddd = pd.DataFrame({
        'TRate (MeV/g/s)':trate,'SRate (MeV/g/s)':srate,'Rate (MeV/g/s)':trate + srate,
        'TDose (MeV/g)':tdose,'SDose (MeV/g)':sdose,'Dose (MeV/g)':tdose + sdose
    })
    ddd[['TDDD (MeV/g)','SDDD (MeV/g)','DDD (MeV/g)']] = ddd[['TDose (MeV/g)','SDose (MeV/g)','Dose (MeV/g)']].cumsum()

    df = pd.concat([orbit['DF'],tflux['DF'],sflux['DF'],gflux['DF'],tid,ddd],axis=1)
    df.to_csv(f'trajectory/segment{segment}.csv',index=False)
    return df,{'orbit':orbit,'tflux':tflux,'sflux':sflux,'gflux':gflux,'tid':tid,'ddd':ddd}

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
    result[['TTID (krad)','STID (krad)','GTID (krad)','TID (krad)']] = result[['TDose (krad)','SDose (krad)','GDose (krad)','Dose (krad)']].cumsum().ffill()
    result[['TDDD (MeV/g)','SDDD (MeV/g)','DDD (MeV/g)']] = result[['TDose (MeV/g)','SDose (MeV/g)','Dose (MeV/g)']].cumsum().ffill()
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
    mission[['TTID (krad)','STID (krad)','GTID (krad)','TID (krad)']] = mission[['TDose (krad)','SDose (krad)','GDose (krad)','Dose (krad)']].cumsum().ffill()
    mission[['TDDD (MeV/g)','SDDD (MeV/g)','DDD (MeV/g)']] = mission[['TDose (MeV/g)','SDose (MeV/g)','Dose (MeV/g)']].cumsum().ffill()
    return mission

def get_atten():
    niel = read_spenvis('trajectory/MISSION/spenvis_nio.txt')
    patten = pd.DataFrame(index=niel[-4]['DF']['Energy (MeV)'])
    for col in niel[-4]['DF'].columns[1:]:
        print(col)
        patten[col] = niel[-4]['DF'][col].values / niel[-4]['DF']['Unshielded'].values
    print(patten)
    patten.to_csv('radstats/proton_atten.csv')

def event(event_file,shielding='0.370 mm'):
    file = open(event_file).read().split('\n')
    energies = []
    check_next = False
    skiprows = 0
    for i,line in enumerate(file):
        if check_next: energies.append(float(line.split('channel')[1].split('-')[0]))
        check_next = False
        if line.startswith('float p'): check_next = True
        if line.startswith('data:'): skiprows = i + 1

    dflux = pd.read_csv(event_file,skiprows=skiprows,index_col='time_tag')
    dflux.index = pd.to_datetime(dflux.index)
    dflux = dflux['1989-10-19 12:00:00' : '1989-11-01']
    atten = pd.read_csv('radstats/proton_atten.csv',index_col='Energy (MeV)')
    atten = atten.loc[[0.63,4.5,9,16,40,90,110]]
    dflux = dflux.drop(columns='e2_flux_i') * atten[shielding].values
    dflux.columns = [f'SEP {x} MeV' for x in energies]
    iflux = pd.DataFrame(index=dflux.index)
    for i,col in enumerate(dflux.columns): iflux[col]=dflux[dflux.columns[i:len(dflux.columns)+1]].sum(axis=1) * energies[i]
    pstar = pd.read_csv('radstats/pstar.txt',sep=' ',skiprows=8,header=None)
    pstar.columns = ['Energy','LET',2]
    pstar = pstar.pivot(columns='Energy',values='LET').fillna(method='bfill').iloc[0]
    dose = pd.DataFrame({'Rate (krad/s)':iflux[iflux > 0].mul(pstar[energies].values).sum(axis=1) * 1.6e-8 / 1000}) # krad
    dose['Dose (krad)'] = iflux.index
    dose['Dose (krad)'] = dose['Rate (krad/s)'] * dose['Dose (krad)'].diff().dt.total_seconds().bfill()
    dose['TID (krad)'] = dose['Dose (krad)'].cumsum()
    env = pd.concat([dflux,dose],axis=1)
    env['Time (s)'] = env.index
    env['Time (s)'] = (env['Time (s)'] - env['Time (s)'].iloc[0]).dt.total_seconds()
    env['Time (hrs)'] = env['Time (s)'] / 3600
    env['SegmentNum'] = 0
    return env

