from turtle import pos
import numpy as np
import pandas as pd
import itertools

decimals = 20

def marginal(probs):
    Pmarg = [None] * (2 ** len(probs))
    for i,truth in enumerate(itertools.product([True,False],repeat=len(probs))):
        Pmarg[i] = np.prod([p if truth[n] else 1 - p for n,p in enumerate(probs)])
    return np.around(Pmarg,decimals)

def OR(probs):
    Pmarg = marginal(probs)
    Pcond = np.ones(len(Pmarg))
    Pcond[-1] = 0
    return np.around(np.dot(Pcond,Pmarg),decimals)

def AND(probs):
    Pmarg = marginal(probs)
    Pcond = np.zeros(len(Pmarg))
    Pcond[0] = 1
    return np.around(np.dot(Pcond,Pmarg),decimals)

def VOTE(probs,k=1):
    Pmarg = marginal(probs)
    Pcond = [sum(i) >= k for i in itertools.product([True,False],repeat=len(probs))]
    return np.around(np.dot(Pcond,Pmarg),decimals)

def propagate(system,element):
    if element.tag == 'ref': element = system.xpath(f'.//node[@name="{element.get("name")}"]')[0]
    if element.get('prob','-1') == '-1':
        p = [float(propagate(system,child)) for child in element.xpath('*')]
        if element.get('gate') == 'OR': element.set('prob',str(OR(p)))
        if element.get('gate') == 'AND': element.set('prob',str(AND(p)))
        if element.get('gate') == 'VOTE': element.set('prob',str(VOTE(p,float(element.get('k')))))
    return element.get('prob')

def faults(system,probs,root=None):
    root = system.getroot() if root is None else system.xpath(f'..//node[@name="{root}"]')[0]
    nodes = root.xpath('.//node') + [root]
    events = root.xpath('.//event')
    refs = root.xpath('.//ref')
    for ref in refs:
        if system.xpath(f'..//node[@name="{ref.get("name")}"]')[0] in nodes: continue
        else: 
            nodes = system.xpath(f'..//node[@name="{ref.get("name")}"]/..//node') + nodes
            events = system.xpath(f'..//node[@name="{ref.get("name")}"]/.//event') + events
    result = pd.DataFrame(index=probs.index,
        columns=[event.get('probs') for event in events] + [node.get('name') for node in nodes])
    for i,t in enumerate(probs.index):
        print(i,end='\r')
        for event in events:
            repair = float(event.get('repair','0').replace('\\infty','-1')) / 3600
            #require = float(event.get('require','0')) / 3600
            if repair == 0: prob = probs[event.get('probs')][t]
            elif repair > 0: prob = OR(probs[event.get('probs')][max(0,t-repair):t])
            elif repair < 0: prob = OR([
                    max(0,result[event.get('probs')].iloc[max(0,i-1)]),
                    probs[event.get('probs')][t]
                ])
            result.loc[t][event.get('probs')] = prob
            event.set('prob',str(prob))
        for node in nodes: node.set('prob','-1')
        propagate(system,root)
        for node in nodes: result[node.get('name')][t] = float(node.get('prob',np.nan))
    print(' '*20,end='\r')
    return result.astype(float)

def importance(system,faults,root=None):
    root = system.getroot() if root is None else system.xpath(f'..//node[@name="{root}"]')[0]
    nodes = root.xpath('.//node') + [root]
    events = root.xpath('.//event')
    refs = root.xpath('.//ref')
    for ref in refs:
        if system.xpath(f'..//node[@name="{ref.get("name")}"]')[0] in nodes: continue
        else: 
            nodes = system.xpath(f'..//node[@name="{ref.get("name")}"]/..//node') + nodes
            events = system.xpath(f'..//node[@name="{ref.get("name")}"]/.//event') + events
    marginal = pd.DataFrame(index=faults.index,columns=faults.columns)
    critical = pd.DataFrame(index=faults.index,columns=faults.columns)
    diagnostic = pd.DataFrame(index=faults.index,columns=faults.columns)
    raw = pd.DataFrame(index=faults.index,columns=faults.columns)
    rrw = pd.DataFrame(index=faults.index,columns=faults.columns)
    for i,t in enumerate(faults.index):
        print(i,end='\r')
        for event in events: event.set('prob',str(faults.loc[t][event.get('probs')]))
        for element in events + nodes:
            if element.tag == 'node': col = element.get('name')
            elif element.tag == 'event': col = element.get('probs')
            element_prob = faults.loc[t][col]
            root_prob = faults.loc[t][root.get('name')]
            for node in nodes: node.set('prob','-1')
            element.set('prob','1')
            prob_true = float(propagate(system,root))
            for node in nodes: node.set('prob','-1')
            element.set('prob','0')
            prob_false = float(propagate(system,root))
            marginal.loc[t][col] = prob_true - prob_false
            critical.loc[t][col] = marginal.loc[t][col] * element_prob / root_prob if root_prob > 0 else 1
            diagnostic.loc[t][col] = element_prob * prob_true / root_prob if root_prob > 0 else 1
            raw.loc[t][col] = prob_true / root_prob if root_prob > 0 else 1
            rrw.loc[t][col] = root_prob / (prob_false if prob_false > 0 else 1e-9)
            element.set('prob',str(element_prob))
    print(' '*20,end='\r')
    return {'marginal':marginal,'critical':critical,'diagnostic':diagnostic,'raw':raw,'rrw':rrw}