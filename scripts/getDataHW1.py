import pandas as pd

def get_data(data_path="../data",testData = False):
    assert os.path.isfile(data_path), f"{os.path.realpath(data_path)} : File not exist"
    
    train_data = pd.read_csv(data_path+"/train_eng.csv", engine='python' ,encoding = "latin-1")
    test_data = pd.read_csv(data_path+"/test_eng.csv", engine='python' ,encoding = "latin-1")

    labels_keys = {value: i for i, (value, count) in enumerate(train_data.Gender.value_counts().items())}
    train_data['Gender'] = train_data['Gender'].apply(lambda x: labels_keys.get(x))
    test_data['Gender'] = test_data['Gender'].apply(lambda x: labels_keys.get(x))
    
    max_len = len(max(train_data.iloc[:,0], key=len))

    for name in range(len(list(train_data.iloc[:,0]))):
      names=''
      for char in range(len(train_data.Name[name])):
        names += train_data.Name[name][char]+' '
      train_data.Name[name] = names
    print(train_data.iloc[:,0])

    for name in range(len(list(test_data.iloc[:,0]))):
      names=''
      for char in range(len(test_data.Name[name])):
        names += test_data.Name[name][char]+' '
      test_data.Name[name] = names
    print(test_data.iloc[:,0])
    
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(train_data['Name'].values, tf.string),
            tf.cast(train_data['Gender'].values, tf.int64)
        ))

    test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(test_data['Name'].values, tf.string),
                tf.cast(test_data['Gender'].values, tf.int64)
            ))
    
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_data, test_data, max_len

if __name__ == '__main__':
    main(None)
