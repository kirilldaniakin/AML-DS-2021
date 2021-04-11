import pandas as pd

def get_data(data_path="../data",testData = False):
    assert os.path.isfile(data_path), f"{os.path.realpath(data_path)} : File not exist"
    
    train_data = pd.read_csv(data_path+"/train_eng.csv", engine='python' ,encoding = "latin-1")
    test_data = pd.read_csv(data_path+"/test_eng.csv", engine='python' ,encoding = "latin-1")

    labels_keys = {value: i for i, (value, count) in enumerate(train_data.Gender.value_counts().items())}
    train_data['Gender'] = train_data['Gender'].apply(lambda x: labels_keys.get(x))
    test_data['Gender'] = test_data['Gender'].apply(lambda x: labels_keys.get(x))
    
    return train_data, test_data
