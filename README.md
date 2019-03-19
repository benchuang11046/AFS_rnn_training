# AFS_rnn_training
A example use AFS2 to training a model with LSTM




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