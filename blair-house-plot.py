from numpy import genfromtxt
from pytz import timezone
import numpy as np
from pandas.util.testing import DataFrame
import pandas

import matplotlib
import matplotlib.dates
import matplotlib.pyplot as plt


def mean_std(b, slices):
    slicedValues = [[rec['value'] for rec in b
                     if rec['time'] >= begin and rec['time'] <= end]
                        for ( begin, end) in slices]
    avgData = np.array([np.average(v) for v in slicedValues]) # arithmetic mean
    stdData = np.array([np.std(v) for v in slicedValues]) # standard deviation
    # stdErrData = np.array([np.std(v) for v in slicedValues]) / np.sqrt(([len(v) for v in slicedValues]))
    return (avgData, stdData)


blairInsideAll = genfromtxt('blair-inside.tsv', dtype=None, names='time,sid,mid,value')
blairOutsideAll = genfromtxt('blair-outside.tsv', dtype=None, names='time,sid,mid,value')
filt = lambda b: b[np.logical_and(b['time'] != 0, b['sid'] == 2)]
blairOutside = filt(blairOutsideAll)
blairInside = filt(blairInsideAll)

begin = blairInside['time'][0]
end = blairInside['time'][-1]
duration = end - begin
slices = [(begin + duration * step / 100, begin + duration * (step + 2) / 100) for step in range(1, 98)]

slicedDates = [begin + (begin - end) / 2 for ( begin, end) in slices] # re-center date in middle of avg window

idx = pandas.to_datetime(slicedDates, unit='s', utc=True)
df1 = DataFrame({'inside': mean_std(blairInside, slices)[0], 'outside': mean_std(blairOutside, slices)[0]}
                , index=idx, columns=['inside', 'outside'])
df1.plot(kind='line')
plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M', tz=timezone("America/New_York")))


def fill_error_tube(b, color):
    (mean, error) = mean_std(b, slices)
    plt.fill_between(df1.index, mean - error, mean + error, color=color)


fill_error_tube(blairInside, [0.5, 0.5, 0.5, 0.5])
fill_error_tube(blairOutside, [0.5, 0.5, 0.5, 0.5])

plt.ylabel("Temperatur [Celsius]", fontsize=20)
plt.xlabel("Time [Hours]", fontsize=20)

#plt.show()
plt.savefig("blair-house-temperature.png", bbox_inches='tight')


