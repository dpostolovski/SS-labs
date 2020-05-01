from datetime import timedelta
from math import sqrt

from matplotlib import pyplot
from pandas import DataFrame, IntervalIndex
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA, ARMA
from sklearn.metrics import mean_squared_error, r2_score
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
    def __init__(self):
        self.params = None
    def predict(self, data, interval):
        small_data = data[-35:-1]

        model = ARMA(small_data, order=(5, 1, 0), freq='h')
        model_fit = model.fit(disp=0, start_params=self.params)
        self.params = model_fit.params
        prediction = model_fit.forecast()

        t_dict = {'prediction': prediction[0][0]}
        return pd.Series(t_dict, name=data.index.max() + interval)

class SarimaxPredictor():
    def __init__(self):
        self.params = None
    def predict(self, data, interval):
        small_data = data[-35:-1]

        model = SARIMAX(small_data,seasonal_order=(2,1,1,24), freq='h', enforce_invertibility=False, enforce_stationarity=False)
        model_fit = model.fit(disp=0, start_params=self.params)
        self.params = model_fit.params
        prediction = model_fit.forecast()

        t_dict = {'prediction': prediction[0]}
        return pd.Series(t_dict, name=data.index.max() + interval)


class Sensor():
    def __init__(self, simulated_weather, predictor_1,predictor_2, threshold_1, threshold_2, interval = timedelta(minutes=60)):
        self.threshold_2 = threshold_2
        self.threshold_1 = threshold_1
        self.observers = list()
        self.simulated_weather = simulated_weather
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2
        self.historical_data = DataFrame()
        self.interval = interval
        self.sent_counter = 0

    def get_measurement(self):
        return self.simulated_weather.get_measurement()

    def run(self):
        measurement = self.get_measurement()
        if self.historical_data.__len__() < 48:
            self.update(measurement)
            self.historical_data = self.historical_data.append(measurement)
            self.sent_counter +=1
            return

        prediction_temp = self.predictor_1.predict(self.historical_data.Temp1, self.interval)
        prediction_rad = self.predictor_2.predict(self.historical_data.Insolation, self.interval)

        if abs(measurement.Temp1.values[0] - prediction_temp.values[0]) > self.threshold_1 or \
            abs(measurement.Insolation.values[0] - prediction_rad.values[0]) > self.threshold_2:

            self.update(measurement)
            self.historical_data = self.historical_data.append(measurement)
            self.sent_counter +=1
        else:
            t_dict = {'Temp1': prediction_temp.values[0],
                      'Insolation': prediction_rad.values[0]}

            self.historical_data = self.historical_data.append(pd.Series(t_dict, name = self.historical_data.index.max() + self.interval))

    def add_observer(self, observer):
        self.observers.append(observer)

    def update(self, current_measurement):
        for observer in self.observers:
            observer.notify(self, current_measurement)


def test_predictor(predictor):
    thresholds_1 = [0.5, 1, 1.5, 2, 2.5]
    thresholds_2 = [25, 50, 75, 100, 150]
    data_sent_list = list()
    errors_temp = list()
    errors_rad = list()

    for i in range(thresholds_1.__len__()):

        simulated_weather = SimulatedWeather(sensor_data)
        sensor = Sensor(simulated_weather, predictor(), predictor(), thresholds_1[i], thresholds_2[i])
        # server = Server()
        # sensor.add_observer(server)
        for i in range(sensor_data.__len__()):
            sensor.run()
        # interpolation = server.interpolate(predictor, 'H')

        error_temp = r2_score(sensor_data.Temp1, sensor.historical_data.Temp1)
        error_insolation_ = r2_score(sensor_data.Insolation, sensor.historical_data.Insolation)

        data_sent = sensor.sent_counter / sensor_data.__len__()
        data_sent_list.append(data_sent)
        errors_temp.append(error_temp)
        errors_rad.append(error_insolation_)

    pyplot.figure(1)
    pyplot.plot(thresholds_1, data_sent_list, label=predictor.__name__)

    pyplot.figure(2)
    pyplot.plot(data_sent_list, errors_temp, label=predictor.__name__ + 'Temperature')
    pyplot.plot(data_sent_list, errors_rad, label=predictor.__name__ + 'Irradiation')

    pyplot.figure(3)
    pyplot.plot(thresholds_1, errors_temp, label=predictor.__name__ + 'Temperature')
    pyplot.plot(thresholds_1, errors_rad, label=predictor.__name__ + 'Irradiation')

    return data_sent_list, errors_temp, errors_rad


if __name__ == '__main__':
    sensor_data = pd.read_hdf('data.hdf', 'sensor')
    sensor_data = sensor_data.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    sensor_data = sensor_data.resample('H').mean()


    simulated_weather = SimulatedWeather(sensor_data)
    sensor = Sensor(simulated_weather, ArimaPredictor(), ArimaPredictor(), 1, 50)
    for i in range(sensor_data.__len__()):
        sensor.run()

    pyplot.plot(sensor_data.Temp1.tolist())
    pyplot.plot(sensor.historical_data.Temp1.tolist(), color='red')
    pyplot.show()

    pyplot.plot(sensor_data.Insolation.tolist())
    pyplot.plot(sensor.historical_data.Insolation.tolist(), color='red')
    pyplot.show()

    result_arima = test_predictor(ArimaPredictor)
    result_sarimax= test_predictor(SarimaxPredictor)

    pyplot.figure(1)
    pyplot.ylabel('% of data sent')
    pyplot.xlabel('Error allowed (degrees C)')
    pyplot.legend()

    pyplot.figure(2)
    pyplot.ylabel('R^2')
    pyplot.xlabel('% of data sent')
    pyplot.legend()

    pyplot.figure(3)
    pyplot.xlabel('Error allowed (degrees C)')
    pyplot.ylabel('R^2')
    pyplot.legend()

    pyplot.show()

    print('finish')