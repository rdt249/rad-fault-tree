import pandas as pd
import numpy as np

hr = 4
alt = 100
timeline = pd.DataFrame({
    'time_s': np.linspace(0,hr*3600,hr*3600),
    'alt_km': [*np.linspace(0,alt,hr*1800), *np.linspace(alt,0,hr*1800)],
    'lat_deg': np.linspace(0,0,hr*3600),
    'lon_deg' : np.linspace(0,0,hr*3600)
}).set_index('time_s')
print(timeline)
timeline.to_csv('HAB/timeline.csv')
