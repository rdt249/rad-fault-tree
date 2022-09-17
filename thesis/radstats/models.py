import numpy as np
from scipy.special import erf
import pandas as pd

def weibull(x,sat,x0,w=1,s=1):
    x = np.array(x)
    y = sat * (1 - np.exp(-np.power((x - x0) / w,s)))
    y[y < 0] = 0
    return y

def lognormal(x,mean,stdev):
    x = np.array(x)
    y = (1/2) * (1 + erf((np.log(x) - mean) / (stdev * np.sqrt(2))))
    return y

example = {
    'GCR':{
        #'SRAM_SEU':{'Saturation':1e-2,'Threshold':1},
        'SRAM_SEFI':{'Saturation':1e-6,'Threshold':10},
        'SRAM_SEL':{'Saturation':1e-7,'Threshold':15},
        'MCU_SEFI':{'Saturation':2e-6,'Threshold':8},
        'MCU_SEL':{'Saturation':2e-7,'Threshold':12},
        'LREG_SEGR':{'Saturation':3e-7,'Threshold':20}
    },
    'Protons':{
        #'SRAM_SEU':{'Saturation':1e-6,'Threshold':10},
        'SRAM_SEFI':{'Saturation':1e-10,'Threshold':50},
        'MCU_SEFI':{'Saturation':2e-10,'Threshold':40},
    },
    'Dose':{
        'SRAM_TID':{'Mean':4,'StDev':0.1},
        'MCU_TID':{'Mean':3.5,'StDev':0.1},
        'LREG_TID':{'Mean':4.5,'StDev':0.1}
    }
}
    
def get_probs(env,events):
    tspec,sspec,gspec = [],[],[]
    for col in env.columns:
        if col.startswith('TRP'): tspec.append(col)
        if col.startswith('SEP'): sspec.append(col)
        if col.startswith('GCR'): gspec.append(col)
    tflux,sflux,gflux,dose = env[tspec].values,env[sspec].values,env[gspec].values,env['TID'].values
    tflux[0] = env[tspec].mean()
    tstep = max(env['Time (s)']) / len(env)
    for rad in events:
        for event in events[rad]:
            events[rad][event]['rate'] = []
            events[rad][event]['prob'] = []
    for i in range(len(env)):
        print(i,end='\r')
        for rad in events:
            for event in events[rad]:
                params = [events[rad][event][key] for key in events[rad][event].keys()][:-2]
                if rad == 'Protons':
                    x = [float(col.split(' ')[1]) for col in tspec] # trapped spectrum
                    model = weibull(x,*params)
                    dflux = tflux[i] / x
                    rate = np.trapz(dflux * model,x)
                    x = [float(col.split(' ')[1]) for col in sspec] # solar spectrum
                    model = weibull(x,*params)
                    dflux = sflux[i] / x
                    rate += np.trapz(dflux * model,x)
                    prob = 1 - np.exp(-rate * tstep)
                if rad == 'GCR':
                    x = [float(col.split(' ')[1]) for col in gspec] # gcr spectrum
                    model = weibull(x,*params)
                    dflux = gflux[i] / x
                    rate = np.trapz(dflux * model,x)
                    prob = 1 - np.exp(-rate * tstep)
                if rad == 'Dose':
                    prob = lognormal(dose[i],*params)
                    rate = -np.log(1 - prob) / tstep
                    if rate == -0: rate = 0
                events[rad][event]['rate'].append(float(rate))
                events[rad][event]['prob'].append(float(prob))
    print(' '*20,end='\r')
    probs = pd.DataFrame(index=env.index)
    names = []
    for rad in events:
        for event in events[rad]:
            names.append(rad.lower()[0] + event)
            probs[names[-1]] = events[rad][event]['prob']
    for name in names:
        effect = name[1:]
        if effect in probs.columns: probs[effect] += probs[name].replace(np.nan,0)
        else: probs[effect] = probs[name].replace(np.nan,0)
        probs = probs.drop(name,axis=1)
    probs.index = env['Time (hrs)']
    return probs