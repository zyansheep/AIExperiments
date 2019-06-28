from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import sys

ts = TimeSeries(key='MHW9HJMN992W0VUK', output_format='pandas')
plt.title('Real-Time Stock Chart')

fig = plt.figure()
ax1 = fig.add_subplot()


def stockchart(symbol):
    data, meta_data = ts.get_intraday(symbol)

    for key, value in data.iteritems():
        print(key)
    plt.plot(data['4. close'])
    plt.show()

stockchart('goog')

for i in range(10)