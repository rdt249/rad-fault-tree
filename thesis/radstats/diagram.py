import schemdraw
from schemdraw import flow,logic,elements,segments
from PIL import Image
import pandas as pd
import numpy as np

from radstats import bayes

decimals = 4
spacing = (3,3.5)

class PAND(elements.Element):
    def __init__(self, dot=False, label=False, *d, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments = logic.And(**kwargs).segments
        self.anchors = logic.And(**kwargs).anchors
        self.segments.append(segments.Segment([(0.35,0.5),(1.5,0)]))
        self.segments.append(segments.Segment([(0.35,-0.5),(1.5,0)]))
        if label: self.segments.append(segments.SegmentText((0.5,0),'PAND',fontsize=35))
        if dot: self.segments.append(segments.SegmentCircle((0.5,-0.3),0.02,fill=True))

class SPARE(elements.Element):
    def __init__(self, dot=False, label=False, *d, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments = logic.And(**kwargs).segments[1:]
        self.anchors = logic.And(**kwargs).anchors
        self.segments.append(segments.SegmentPoly([(0.35,0.75),(1.5,0.75),(1.5,-0.75),(0.35,-0.75)]))
        self.segments.append(segments.SegmentPoly([(0.35,0.75),(0.9,0.75),(0.9,0),(0.35,0)]))
        self.segments.append(segments.Segment([(1.1,0.75),(1.1,-0.75)]))
        if label: self.segments.append(segments.SegmentText((1.25,0),'SPARE',fontsize=35))

class FDEP(elements.Element):
    def __init__(self, dot=False, label=False, *d, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments = logic.And(**kwargs).segments[1:]
        self.anchors = logic.And(**kwargs).anchors
        self.segments.append(segments.SegmentPoly([(0.35,0.75),(1.5,0.75),(1.5,-0.75),(0.35,-0.75)]))
        self.segments.append(segments.SegmentPoly([(0.35,-0.75),(1.1,-0.75),(0.725,0)]))
        self.segments.append(segments.Segment([(1.1,0.75),(1.1,-0.75)]))
        self.segments.append(segments.Segment([(0.725,-0.75),(0.725,-1)]))
        self.anchors['trig'] = (0.725,-1)
        if label: self.segments.append(segments.SegmentText((1.25,0),'FDEP',fontsize=35))

class SEQ(elements.Element):
    def __init__(self, dot=False, label=False, *d, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments = logic.And(**kwargs).segments[1:]
        self.anchors = logic.And(**kwargs).anchors
        self.segments.append(segments.SegmentPoly([(0.35,0.75),(1.5,0.75),(1.5,-0.75),(0.35,-0.75)]))
        self.segments.append(segments.Segment([(1.1,0.75),(1.1,-0.75)]))
        if label: self.segments.append(segments.SegmentText((1.25,0),'SEQ',fontsize=35))
        if dot: self.segments.append(segments.SegmentCircle((0.5,-0.6),0.02,fill=True))

class VOTE(elements.Element):
    def __init__(self, k='k', *d, **kwargs):
        super().__init__(*d, **kwargs)
        self.segments = logic.Or(**kwargs).segments
        self.anchors = logic.Or(**kwargs).anchors
        self.segments.append(segments.SegmentText((1,0),k,fontsize=50))

pos = {}

def get_pos(system,element,mode):
    global pos
    if element.tag == 'ref': element = system.xpath(f'..//node[@name="{element.get("name")}"]')[0]
    if element not in pos.keys():
        x = [get_pos(system,child,mode=mode)[0] for child in element]
        y = [get_pos(system,child,mode=mode)[1] for child in element]
        if mode == 'FT': pos[element] = (sum(x) / len(x), max(y) + spacing[1])
        elif mode == 'BN': pos[element] = (sum(x) / len(x), min(y) - spacing[1])
    return pos[element]

def get_prob(system,element):
    if element.tag == 'ref': element = system.xpath(f'..//node[@name="{element.get("name")}"]')[0]
    if element.get('prob','-1') == '-1':
        p = [float(get_prob(system,child)) for child in element]
        if element.get('gate') == 'OR': element.set('prob',str(bayes.OR(p)))
        if element.get('gate') == 'AND': element.set('prob',str(bayes.AND(p)))
        if element.get('gate') == 'VOTE': element.set('prob',str(bayes.VOTE(p,float(element.get('k')))))
    # print(element.get('name'),element.get('prob'))
    return element.get('prob')

def newFT(system,root=None,file='diagram.png',probs=None,frame=-1,show=False,scale=2):
    root = system.getroot() if root is None else system.xpath(f'..//node[@name="{root}"]')[0]
    nodes = root.xpath('.//node') + [root]
    events = root.xpath('.//event')
    refs = root.xpath('.//ref')
    for ref in refs:
        if system.xpath(f'..//node[@name="{ref.get("name")}"]')[0] in nodes: continue
        else: 
            nodes = system.xpath(f'..//node[@name="{ref.get("name")}"]/..//node') + nodes
            events = system.xpath(f'..//node[@name="{ref.get("name")}"]/.//event') + events
    global pos
    pos = {}
    schemdraw.config(inches_per_unit=scale,fontsize=15*scale,lw=scale)
    for i,event in enumerate(system.xpath('..//event')):
        pos[event] = (i * spacing[0],-spacing[1] * len(event.xpath('ancestor::node')))
    get_pos(system,root,mode='FT')
    initial = bayes.decimals
    bayes.decimals = decimals
    if probs is not None:
        for element in nodes + events: element.set('prob','-1')
        for event in events: event.set('prob',str(np.around(probs[event.get('name')].iloc[frame],decimals)))
        get_prob(system,root)
    bayes.decimals = initial
    with schemdraw.Drawing(file=file,show=show) as d:
        for event in events:
            fill = (1,1,1,0)
            label = event.get('name')
            if probs is not None:
                label += '\np=' + event.get('prob')
                p = float(event.get('prob'))
                fill = (1,1-p/2,1-p/2,p/2)
            d += flow.Circle(fill=fill).anchor('N').at(pos[event]).label(label)
        for node in nodes:
            fill = (1,1,1,0)
            label = node.get('name')
            if probs is not None:
                label += '\np=' + node.get('prob')
                p = float(node.get('prob'))
                fill = (1,1-p/2,1-p/2,p/2)
            if node.get('gate',None) == 'OR':
                d += (new := flow.Box(h=1,fill=fill).anchor('N').at(pos[node]).label(label))
                d += (new := logic.Or(inputs=1,fill=fill).at(new.S).down().reverse())
            if node.get('gate',None) == 'AND':
                d += (new := flow.Box(h=1,fill=fill).anchor('N').at(pos[node]).label(label))
                d += (new := logic.And(inputs=1,fill=fill).at(new.S).down().reverse())
            if node.get('gate',None) == 'VOTE':
                d += (new := flow.Box(h=1,fill=fill).anchor('N').at(pos[node]).label(label))
                d += (new := VOTE(inputs=1,k=node.get('k'),fill=fill).at(new.S).down().reverse())
            for sub in node.xpath('*'):
                if sub.tag == 'ref': sub = system.xpath(f'..//node[@name="{sub.get("name")}"]')[0]
                d += flow.Wire('-|').at(new.in1).to(pos[sub])
    image = Image.open(file)
    image.crop(image.getbbox())
    return image

def FT(system,root=None,file='diagram.png',probs=False,gates=True,times=True,info=False):
    root = system.getroot() if root is None else system.xpath(f'..//node[@name="{root}"]')[0]
    nodes = root.xpath('.//node') + [root]
    events = root.xpath('.//event')
    refs = root.xpath('.//ref')
    for ref in refs:
        if system.xpath(f'..//node[@name="{ref.get("name")}"]')[0] in nodes: continue
        else: 
            nodes = system.xpath(f'..//node[@name="{ref.get("name")}"]/..//node') + nodes
            events = system.xpath(f'..//node[@name="{ref.get("name")}"]/.//event') + events
    global pos
    pos = {}
    if info: schemdraw.config(inches_per_unit = 2,fontsize=30)
    else: schemdraw.config(inches_per_unit = 2,fontsize=30)
    for i,event in enumerate(system.xpath('..//event')):
        pos[event] = (i * spacing[0],-spacing[1] * len(event.xpath('ancestor::node')))
    get_pos(system,root,mode='FT')
    initial = bayes.decimals
    bayes.decimals = decimals
    if type(probs) is pd.DataFrame:
        for event in events: event.set('prob',str(np.around(probs[event.get('name')].iloc[-1],decimals)))
        probs = True
    if probs: get_prob(system,root)
    bayes.decimals = initial
    with schemdraw.Drawing(file=file,show=False) as d:
        for event in events:
            label = event.get('name')
            if info:
                if event.get('model',None) is not None: label += '\n' + event.get('model')
                if event.get('params',None) is not None: label += '\n' + event.get('params')
            if probs: label += '\np=' + event.get('prob')
            if event.get('repair',None) is None or times == False: text = ''
            else: text = f'$t_R={event.get("repair","")}$'
            d += flow.Circle().anchor('N').at(pos[event]).label(label).label(text,loc='bot')
        for node in nodes:
            label = node.get('name')
            if info:
                if node.get('pn',None) is not None: label += '\n' + node.get('pn')
            if probs: label += '\np=' + node.get('prob')
            if node.get('gate',None) == 'OR':
                d += (new := flow.Box(h=1).anchor('N').at(pos[node]).label(label))
                if gates: d += (new := logic.Or(inputs=1).at(new.S).down().reverse())
                else: d += elements.Gap().at(new.absanchors['S']).down().label('OR')
            if node.get('gate',None) == 'AND':
                d += (new := flow.Box(h=1).anchor('N').at(pos[node]).label(label))
                if gates: d += (new := logic.And(inputs=1).at(new.S).down().reverse())
                else: d += elements.Gap().at(new.absanchors['S']).down().label('AND')
            if node.get('gate',None) == 'VOTE':
                d += (new := flow.Box(h=1).anchor('N').at(pos[node]).label(label))
                if gates: d += (new := VOTE(inputs=1,k=node.get('k')).at(new.S).down().reverse())
                else: d += elements.Gap().at(new.absanchors['S']).down().label('VOTE'+node.get('k'))
            for sub in node.xpath('*'):
                if sub.tag == 'ref': sub = system.xpath(f'..//node[@name="{sub.get("name")}"]')[0]
                if gates: d += flow.Wire('-|').at(new.in1).to(pos[sub])
                else: d += flow.Wire('-',arrow='<-').at(new.absanchors['S']).to(pos[sub])
    image = Image.open(file)
    image.crop(image.getbbox())
    image.save(file)
    return image

def BN(system,root=None,file='diagram.png',probs=True,gates=False,times=True):
    root = system.getroot() if root is None else system.xpath(f'..//node[@name="{root}"]')[0]
    nodes = root.xpath('.//node') + [root]
    events = root.xpath('.//event')
    refs = root.xpath('.//ref')
    for ref in refs:
        if system.xpath(f'..//node[@name="{ref.get("name")}"]')[0] in nodes: continue
        else: 
            nodes = system.xpath(f'..//node[@name="{ref.get("name")}"]/..//node') + nodes
            events = system.xpath(f'..//node[@name="{ref.get("name")}"]/.//event') + events
    global pos
    pos = {}
    schemdraw.config(inches_per_unit = 2,fontsize=45)
    levels = 0
    for i,event in enumerate(events):
        if len(event.xpath('ancestor::node')) > levels: levels = len(event.xpath('ancestor::node'))
    for i,event in enumerate(events):
        pos[event] = (i * spacing[0],-spacing[1] * (levels - len(event.xpath('ancestor::node'))))
    get_pos(system,root,mode='BN')
    initial = bayes.decimals
    bayes.decimals = decimals
    if probs: get_prob(system,root)
    bayes.decimals = initial
    with schemdraw.Drawing(file=file,show=False) as d:
        for event in events:
            name = event.get('name')
            if probs: name += '\n' + event.get('prob')
            if event.get('repair',None) is None or times == False: text = ''
            else: text = f'$t_R={event.get("repair","")}$'
            d += flow.State().anchor('S').at(pos[event]).label(name).label(text,loc='top')
        for node in nodes:
            name = node.get('name')
            if probs: name += '\n' + node.get('prob')
            if node.get('gate',None) == 'OR':
                d += (new := flow.State().anchor('S').at(pos[node]).label(name))
                if gates: d += (new := logic.Or(inputs=1).at(new.N).up().reverse())
                else: d += elements.Gap().at(new.absanchors['N']).up().label('OR')
            if node.get('gate',None) == 'AND':
                d += (new := flow.State().anchor('S').at(pos[node]).label(name))
                if gates: d += (new := logic.And(inputs=1).at(new.N).up().reverse())
                else: d += elements.Gap().at(new.absanchors['N']).up().label('AND')
            if node.get('gate',None) == 'VOTE':
                d += (new := flow.State().anchor('S').at(pos[node]).label(name))
                if gates: d += (new := VOTE(inputs=1,k=node.get('k')).at(new.N).up().reverse())
                else: d += elements.Gap().at(new.absanchors['N']).up().label('VOTE'+node.get('k'))
            for sub in node.xpath('*'):
                if sub.tag == 'ref': sub = system.xpath(f'..//node[@name="{sub.get("name")}"]')[0]
                if gates: d += flow.Wire('-').at(pos[sub]).to(new.in1)
                else: d += flow.Wire('-',arrow='->').at(pos[sub]).to(new.absanchors['N'])
    image = Image.open(file)
    image.crop(image.getbbox())
    image.save(file)
    return image

def events(parts):
    print(parts)