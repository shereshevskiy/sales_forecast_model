import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from datetime import datetime, timedelta


class SalesForecast:

    def __init__(self, data, params, target_col="target"):
        self.data = data.copy()
        self.params = params

        self.target_col = target_col

        self.model = None
        self.feature_list = None
        self.data_with_forecast = None

    def _add_features(self, data):
        # weekly seasonality
        k_season_series = self.params["k_season_series"]
        hours_in_week = 168  # number of hours of the week
        for i in k_season_series:
            data[f'weekly_SIN_{i}'.replace(".", "_")] = np.sin(
                (np.arange(len(data)) + 1) * 2 * np.pi * i / hours_in_week)
            data[f'weekly_COS_{i}'.replace(".", "_")] = np.cos(
                (np.arange(len(data)) + 1) * 2 * np.pi * i / hours_in_week)

        # конец и начало месяца
        data['month_end'] = data.index.is_month_end.astype(int)
        data['month_start'] = data.index.is_month_start.astype(int)

        data["weekday"] = data.index.weekday
        data['is_weekend'] = data.weekday.isin([5, 6]).astype(int)

        data['is_even_week'] = ((data.index.isocalendar().week % 2).astype(int) == 0).astype(int)

        self.feature_list = data.drop(self.target_col, axis=1).columns

        return data

    def _fit(self, data):
        params = self.params
        p = params['p']
        d = params['d']
        q = params['q']
        P = params['P']
        D = params['D']
        Q = params['Q']
        S = params['S']

        target_col = self.target_col
        feature_list = self.feature_list

        model = None
        try:
            model = sm.tsa.statespace.SARIMAX(endog=data[target_col],
                                              exog=data[feature_list],
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, S)).fit()
        except ValueError:
            print('wrong parameters (ValueError):', params)
        except LinAlgError:
            print('wrong parameters (LinAlgError):', params)
        except MemoryError as error:
            print(f"MemoryError message: {error}")
            print('wrong parameters (MemoryError):', params)
        except Exception as exception:
            print(f"Exception message: {exception}")
            print('wrong parameters (Exception):', params)

        return model

    def fit(self):
        data = self._add_features(self.data)
        self.model = self._fit(data)

    def _get_future_template(self, prediction_horizon):
        data = self.data
        start_future = (data.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')  # начальная дата прогноза
        end_future = (datetime.strptime(start_future, '%Y-%m-%d %H:%M:%S') +
                      (prediction_horizon - 1) * timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')  # конечная дата
        self.date_list = pd.date_range(start_future, end_future, freq="D")  # временной ряд для прогноза
        data_draft = data[[self.target_col]].copy()
        future = pd.DataFrame(index=self.date_list, columns=data_draft.columns)  # заготовка для прогноза
        return future, data_draft

    def _get_data_with_forecast(self, prediction_horizon):
        data = self.data
        future_template, data_draft = self._get_future_template(prediction_horizon)

        future_template = self._add_features(future_template)
        data_with_forecast = pd.concat([data_draft, future_template], sort=True)

        start = len(data)
        end = len(data) + len(self.date_list) - 1

        exog_future = future_template.loc[self.date_list, self.feature_list]

        data_with_forecast['forecast'] = np.round(self.model.predict(start=start, end=end, exog=exog_future), 1).clip(0)

        return data_with_forecast

    def predict(self, prediction_horizon):
        self.data_with_forecast = self._get_data_with_forecast(prediction_horizon)
        forecast = self.data_with_forecast[['forecast']].tail(prediction_horizon).copy()
        return forecast

    def get_data_with_forecast(self):
        return self.data_with_forecast
