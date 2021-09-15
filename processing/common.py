import os

import pandas as pd

from settings import train_file, data_path


def read_data(store_id, file_name=train_file, path=data_path):
    data = pd.read_csv(os.path.join(path, file_name), index_col=1, parse_dates=True)
    data = data[data.id == store_id].drop("id", axis=1)
    return data


def fill_vacation(data):
    ind = data.index
    full_ind = pd.date_range(ind[0], ind[-1], freq='D')
    data = pd.DataFrame(data, index=full_ind)
    data = data.fillna(method="ffill")
    return data


def data_preprocessing(data, store_id):
    data = fill_vacation(data)
    if store_id in [4, 8, 10, 13, 17]:
        data = drop_days(data)
    return data


def drop_days(data, num_dropped_days=224):
    """
    Deletes days where data has been missing for 7 months.
    For correct loading into the model, the dates are shifted to be continuous.
    In this case, the days of the week and the parity of the week number are preserved.
    This is achieved by the correct number of deleted days
    (equal to 224) - a multiple of 7 and the number of weeks is even: 224/7 = 32
    """
    data = data.copy()
    dropped_date = pd.date_range("2016-06-01", periods=num_dropped_days, freq="D")
    data = data.drop(dropped_date)
    ind = pd.date_range(end=data.index[-1], periods=len(data))
    data.index = ind
    return data


def data_split(data, num_days=31):
    train, valid = data[:-num_days].copy(), data[-num_days:].copy()
    return train, valid


def calc_sMAE(predict, valid, train):
    sMAE = (predict["forecast"] - valid["target"]).abs().mean() / train["target"].abs().mean()
    return sMAE
