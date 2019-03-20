# AFS_rnn_training
A example use AFS2 to training a model with LSTM

## Description
Use RNN method to inference the future temperature with the sensor feature.

* Generate data sets from past data, parameter `look_back` can set how many amount of past data to predict.
* Use Keras backend Tensorflow, and implement RNN with LSTM.
* The Neural Network architecture is following:

```
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

```

### model_para
|Name         |Type           |Description|
|-------------|---------------|-----------|
|epoch         |Integer       |Training epoch, `default=100`|
|LSTM_unit     |Integer       |Dimensionality of the output space, `default=16`|
|look_back     |Integer       |The step to look back, `default=12`|
|model_name    |String        |Model repository name, `default=rnn_model.h5`|



## APM firehose information
APM firehose information set by portal, and get from environment variable. To set the following code in notebook to test.

```
os.environ['PAI_DATA_DIR'] = """{
    "type": "apm-firehose",
    "data": {
        "username": "*****@gmail.com",
        "password": "*****",
        "apmUrl": "https://api-apm-adviotsense-demo-training.wise-paas.com",
        "timeRange": [],
        "timeLast": {},
        "job_config": {},
        "resultProfile": "ben_machine",
        "parameterList": ["pressure", "temperature"],
        "machineIdList": [221]
    }
}"""
```