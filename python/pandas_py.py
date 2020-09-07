#!/usr/bin/env python3

"""
Pandas tutorial
- how to read from csv
- how to change DataFrame column name(s)
- boxplot per group ans save figure
- describe (meta statistic information)
- groupby category & per-group hitogram

=======================================
  Data Format
=======================================
Sample#    Payload(Bytes)   Latency(us)
1          16               2.1
2          16               4.4
...
101        64               3.3
...
"""

import pandas as pd
import os
import matplotlib.pyplot as plt


print('Pandas Example')
# Read from CSV file
f_data = './pandas_data.csv'
file_base = os.path.splitext(os.path.basename(f_data))[0]
df = pd.read_csv(f_data)
# change DataFrame column name
df = df.rename(columns={'Payload [Bytes]': 'Payload', 'Latency [us]': 'Latency'})
grouped = df.groupby('Payload')

# describe
# describe: 50%, 90%, 99%, 99.99% percentile
# to_string: format to 3 decimals
print(grouped['Latency'].describe(percentiles=[0.5, 0.9, 0.99, 0.9999]).to_string(float_format='%.3f'))

# boxplot, save then close figure
ax = df.boxplot(column='Latency', by='Payload', rot=45)
ax.set_ylabel('Latency (us)')
ax.set_title('{}'.format(file_base))
ax.figure.savefig('pandas_boxplot.svg')
plt.close(ax.get_figure())

# per group analysis histogram
for name, group in grouped:
    print('Analyzing: {}'.format(name))
    desp = group['Latency'].describe(percentiles=[0.5, 0.9])
    print(desp)
    ax = group['Latency'].plot.hist(100)
    ax.set_xlabel('Latency (us)')
    ax.set_title('Histogram: payload ({})'.format(name))
    ax.grid()
    ax.text(ax.get_xlim()[1]*0.5, ax.get_ylim()[1]*0.4,
            'Count:{}\nMean:{:.3f}\nStd:{:.3f}\nMin:{:.3f}\n50%:{:.3f}\nMax:{:.3f}'.format(
                len(group), desp['mean'], desp['std'], desp['min'], desp['50%'], desp['90%'], desp['max']
            ))
    plt.show()
