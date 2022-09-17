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

def propagate(element):
    if element.get('prob','-1') == '-1':
        p = [float(propagate(child)) for child in element.xpath('*')]
        if element.get('gate') == 'OR': element.set('prob',str(OR(p)))
        if element.get('gate') == 'AND': element.set('prob',str(AND(p)))
        if element.get('gate') == 'VOTE': element.set('prob',str(VOTE(p,float(element.get('k')))))
    return element.get('prob')

def faults(system,probs):
    nodes = system.xpath('..//node')
    nodes = nodes[1:] + [nodes[0]]
    events = system.xpath('..//event')
    result = pd.DataFrame(
        columns=[event.get('probs') for event in events] + [node.get('name') for node in nodes],
        index=probs.index
    )
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
        propagate(system.getroot())
        for node in nodes: result[node.get('name')][t] = float(node.get('prob',np.nan))
    print(' '*20,end='\r')
    return result

def importance(system,faults):
    nodes = system.xpath('..//node')
    nodes = nodes[1:] + [nodes[0]]
    events = system.xpath('..//event')
    marginal = pd.DataFrame(index=faults.index,
        columns=[event.get('probs') for event in events] + [node.get('name') for node in nodes])
    for i,t in enumerate(faults.index):
        print(i,end='\r')
        for event in events: event.set('prob',str(faults.loc[t][event.get('probs')]))
        for element in events + nodes:
            initial = element.get('prob')
            for node in nodes: node.set('prob','-1')
            element.set('prob','1')
            prob_true = propagate(system.getroot())
            for node in nodes: node.set('prob','-1')
            element.set('prob','0')
            prob_false = propagate(system.getroot())
            if element.tag == 'node': col = element.get('name')
            elif element.tag == 'event': col = element.get('probs')
            marginal.loc[t][col] = float(prob_true) - float(prob_false)
            element.set('prob',initial)
    print(' '*20,end='\r')
    return marginal