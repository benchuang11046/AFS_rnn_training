
## Environment variable setting. Disable these when running task.
import os

## The APP variable from environment variable. 
os.environ['model_para'] = """{
        "epoch": 100,
        "LSTM_unit": 16,
        "look_back": 12,
        "model_name": "rnn_model.h5"
}"""

## APM firehose information set by portal, and get from environment variable. To set the following code in notebook to test.
# os.environ['PAI_DATA_DIR'] = """{
#     "type": "apm-firehose",
#     "data": {
#         "username": "*****@gmail.com",
#         "password": "*****",
#         "apmUrl": "https://api-apm-adviotsense-demo-training.wise-paas.com",
#         "timeRange": [],
#         "timeLast": {},
#         "job_config": {},
#         "resultProfile": "ben_machine",
#         "parameterList": ["pressure", "temperature"],
#         "machineIdList": [221]
#     }
# }"""


###############Training code###############


import requests
import pandas as pd
import requests.packages.urllib3
from datetime import datetime, timedelta
import numpy as np
import math, json, os
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from afs import models
requests.packages.urllib3.disable_warnings()


global apmUrl
global sso_username, sso_password
global resultProfile, feature_list, machineIdList

PAI_DATA_DIR = json.loads(os.getenv('PAI_DATA_DIR',{}))
apmUrl = PAI_DATA_DIR['data']['apmUrl']
sso_username = PAI_DATA_DIR['data']['username']
sso_password = PAI_DATA_DIR['data']['password']
resultProfile = PAI_DATA_DIR['data']['resultProfile']
feature_list = PAI_DATA_DIR['data']['parameterList']
machineIdList = PAI_DATA_DIR['data']['machineIdList']


def read_apm_data():
    # Connection Information
    payload = dict()
    payload['username'] = sso_username
    payload['password'] = sso_password

    # Get Token through SSO Login
    resp_sso = requests.post('https://portal-sso.wise-paas.com/v2.0/auth/native', 
                     json=payload,
                     verify=False)
    header = dict()
    header['content-type'] = 'application/json'
    header['Authorization'] = 'Bearer ' + resp_sso.json()['accessToken']

    # HIST_RAW_DATA API docs
    # https://portal-apmapidoc-acniotsense-apmdemo.wise-paas.com.cn/#api-Data-RGetHistRawData
    APM_HIST = apmUrl + '/hist/raw/data'

    now = datetime.now()
    past = now - timedelta(days=30)
    Query_to = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    Query_from = past.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # Get node and feature to dataframe, and concat them.
    dataframe_list = []
    for apm_nodeid in machineIdList:
        for feature in feature_list:
            payload = dict()
            payload['nodeId'] = apm_nodeid
            payload['sensorType'] = 'monitor'
            payload['sensorName'] = feature
            payload['startTs'] = Query_from
            payload['endTs'] = Query_to

            resp_apm_raw = requests.get(APM_HIST, 
                                        params=payload, 
                                        headers=header,
                                        verify=False)
            df = pd.read_json(
                str(json.dumps(resp_apm_raw.json()['value'])), orient='records')
            df = df.set_index(
                pd.DatetimeIndex(df['ts'])).sort_index(
                ascending=True).drop(
                columns='ts').rename(
                columns={'v': feature})
            dataframe_list.append(df)
    feature_df = pd.concat(dataframe_list, axis=1, sort=False)
    
    print('feature_df:', feature_df)
    return feature_df[[feature_list[0]]]


def create_dataset(dataset, look_back):

    dataX, dataY = [], []
    # dataX is feature, dataY is label
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)


def load_dataset():

    model_para = config()['model_para']

    # load the dataset
    dataframe = read_apm_data()
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    raw_data = dataset

    # normalize the dataset from 0 to 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, model_para['look_back'])
    testX, testY = create_dataset(test, model_para['look_back'])

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX, trainY, testX, testY, scaler


def create_rnn_model():

    model_para = config()['model_para']

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(model_para['LSTM_unit'], input_shape=(1, model_para['look_back'])))
    model.add(Dense(units=128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def train_and_test():

    model_para = config()['model_para']

    trainX, trainY, testX, testY, scaler = load_dataset()
    model = create_rnn_model()
    model.fit(trainX, trainY, epochs=model_para['epoch'], batch_size=5, verbose=0)
    model.summary()

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train RMSE: %.2f ' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test RMSE: %.2f ' % (testScore))

    # colculate training accuracy 
    label = trainY[0]
    predict = trainPredict[:,0]

    # save and upload model
    model.save(model_para['model_name'])
    evaluation_result = {"TestRMSE": testScore, 
                         "TrainRMSE": trainScore}
    tags = {"machine": resultProfile}

    afs_models = models()
    afs_models.upload_model(model_path=model_para['model_name'], loss=trainScore, 
                            tags=tags, extra_evaluation=evaluation_result,
                            model_repository_name=model_para['model_name'])
    return evaluation_result


def config():
    return {'model_para': json.loads(os.getenv('model_para', None))}


if __name__ == '__main__':
    model_para = config()['model_para']
    resp={}
    results = train_and_test()
    resp.update({'model_para': config()['model_para']})
    resp.update({'results': results})
    print(resp)
         
                     