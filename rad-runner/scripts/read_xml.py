from xml.etree import ElementTree
import numpy as np

system = ElementTree.parse('HAB/system.xml').getroot()[0]

for box in system.iter('box'):
    box.set('fault',-1)

for part in system.iter('part'):
    part.set('fault',np.random.rand() < float(part.get('risk')))

def check(box):
    if box.get('fault') == -1:
        faults = [check(child) for child in list(box)]
        if box.get('fx') == 'OR': 
            box.set('fault',any(faults))
        elif box.get('fx') == 'AND':
            box.set('fault',all(faults))
    return box.get('fault')

check(system)