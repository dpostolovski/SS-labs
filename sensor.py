from datetime import timedelta

from matplotlib import pyplot
from pandas import DataFrame, IntervalIndex
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA, ARMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Server():
    def __init__(self):
        self.historical_data = pd.DataFrame()

    def notify(self, observable, data):
        self.historical_data = self.historical_data.append(data)

    def interpolate(self, predictor, freq):

        idx = pd.period_range(min(self.historical_data.index), max(self.historical_data.index), freq=freq)
        result = DataFrame(index=idx.to_timestamp())
        result['Temp1'] = np.NaN
        result.update(self.historical_data)
        X = result['Temp1']
        for i in range(result.__len__()):
            if np.isnan(X[ result.index[i] ]):
                prediction = predictor.predict( result[:i], interval=timedelta(minutes=5) )
                X[result.index[i]] = prediction
        #predictor.predict()
        return X



class SimulatedWeather():
    def __init__(self, data):
        self.data = data
        self.counter =0

    def get_measurement(self):
        self.counter += 1
        return self.data.iloc[[self.counter - 1]]


class Predictor():
    def predict(self, data, interval):
        prediction = 0
        t_dict = { 'Temp1' : prediction }
        return pd.Series(t_dict, name=data.index.max() + interval)

class ArimaPredictor():
    def predict(self, data, interval):
        model = ARIMA(data.Temp1, order=(5, 1, 0), freq='H')
        model_fit = model.fit(disp=0)
        prediction = model_fit.forecast()

        t_dict = {'Temp1': prediction[0][0]}
        return pd.Series(t_dict, name=data.index.max() + interval)

class ArmaPredictor():
    def predict(self, data, interval):
        model = SARIMAX(data.Temp1,order=(5, 1, 0), freq='H')
        model_fit = model.fit(disp=0)
        prediction = model_fit.forecast()

        t_dict = {'Temp1': prediction[0]}
        return pd.Series(t_dict, name=data.index.max() + interval)


class Sensor():
    def __init__(self, simulated_weather, predictor, threshold, interval = timedelta(minutes=60)):
        self.observers = list()
        self.simulated_weather = simulated_weather
        self.predictor = predictor
        self.threshold= threshold
        self.historical_data = DataFrame()
        self.interval = interval
        self.sent_counter = 0

    def get_measurement(self):
        return self.simulated_weather.get_measurement()

    def run(self):
        measurement = self.get_measurement()
        if self.historical_data.__len__() < 15:
            self.update(measurement)
            self.historical_data = self.historical_data.append(measurement)
            return

        prediction = self.predictor.predict(self.historical_data, self.interval)

        if abs(measurement.Temp1.values[0] - prediction.values[0]) > self.threshold:
            self.update(measurement)
            self.historical_data = self.historical_data.append(measurement)
            self.sent_counter +=1
        else:
            self.historical_data = self.historical_data.append(prediction)

    def add_observer(self, observer):
        self.observers.append(observer)

    def update(self, current_measurement):
        for observer in self.observers:
            observer.notify(self, current_measurement)


def test_predictor(predictor):
    thresholds = [0.5, 1, 1.5, 2, 2.5]
    data_sent_list = list()
    errors = list()

    for th in thresholds:

        simulated_weather = SimulatedWeather(temperature_data)
        sensor = Sensor(simulated_weather, predictor, th)
        # server = Server()
        # sensor.add_observer(server)
        for i in range(temperature_data.__len__()):
            sensor.run()
        # interpolation = server.interpolate(predictor, 'H')
        error = mean_squared_error(temperature_data, sensor.historical_data)
        data_sent = sensor.sent_counter / temperature_data.__len__()
        data_sent_list.append(data_sent)
        errors.append(error)

    pyplot.figure(1)
    pyplot.plot(thresholds, data_sent_list, label=predictor.__class__.__name__)

    pyplot.figure(2)
    pyplot.plot(data_sent_list, errors, label=predictor.__class__.__name__)

    pyplot.figure(3)
    pyplot.plot(thresholds, errors, label=predictor.__class__.__name__)


if __name__ == '__main__':
    temperature_data = pd.read_hdf('temperature.hdf', 'temperature')
    temperature_data = temperature_data.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    temperature_data = temperature_data.resample('H').mean()
    temperature_data = temperature_data[:200]

    simulated_weather = SimulatedWeather(temperature_data)
    sensor = Sensor(simulated_weather, ArmaPredictor(), 1)
    for i in range(temperature_data.__len__()):
        sensor.run()

    pyplot.plot(temperature_data.Temp1.tolist())
    pyplot.plot(sensor.historical_data.Temp1.tolist(), color='red')
    pyplot.show()

    test_predictor(ArimaPredictor())
    test_predictor(ArmaPredictor())

    pyplot.figure(1)
    pyplot.ylabel('% of data sent')
    pyplot.xlabel('Error allowed (degrees C)')
    pyplot.legend()

    pyplot.figure(2)
    pyplot.ylabel('MSE')
    pyplot.xlabel('% of data sent')
    pyplot.legend()

    pyplot.figure(3)
    pyplot.xlabel('Error allowed (degrees C)')
    pyplot.ylabel('MSE')
    pyplot.legend()

    pyplot.show()