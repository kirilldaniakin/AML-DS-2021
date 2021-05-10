import numpy as np
from context import scripts
import scripts
import datetime
# Import NumPy and PyTorch
import numpy as np
import torch

# Import PyTorch Ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.metrics import Loss
from ignite.metrics import MeanSquaredError

# Import Tensorboard
from tensorboardX import SummaryWriter

# Import Utility Functions
#from loader import Loader
from datetime import datetime

# Import the Model Script
from Models import *
from loader import Loader

# %load_ext tensorboard
if __name__ == '__main__':
    # get data and max name length
    # Setup TensorBoard logging
    log_dir = 'runs/simple_mf_01_' + str(datetime.now()).replace(' ', '_')
    writer = SummaryWriter(log_dir=log_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_ratings, df_ratings_test = scripts.get_data(data_path="../data/cf")

    R = df_ratings.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0).to_numpy()

    mapping = df_ratings['movieId'].unique()
    map_m = {v: k for k, v in dict(enumerate(mapping)).items()}
    mapping = df_ratings['userId'].unique()
    map_u = {v: k for k, v in dict(enumerate(mapping)).items()}
    df_ratings_test=df_ratings_test.replace({"movieId": map_m})
    df_ratings_test=df_ratings_test.replace({"userId": map_u})
    test_x = df_ratings_test[['userId', 'movieId']].to_numpy().astype(np.int64)
    test_y = df_ratings_test['rating'].to_numpy()
    R = torch.from_numpy(R).to(device)
    
    n_user = R.shape[0]
    n_item = R.shape[1]

    mf = MF(R, n_user, n_item, writer=writer, k=10, c_vector=1e-6).to(device)
    mf.load_state_dict(torch.load("mf.pt"))
    
    R_test = df_ratings_test.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0).to_numpy()
    
    R_test = torch.from_numpy(R_test).to(device)
    
    # Create a supervised evaluator
    def validation_step(engine, batch):
        mf.eval()
        with torch.no_grad():
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = mf(x)
            loss = mf.loss(x, y_pred, y, R_test)
            #print(loss.item())
            return loss.item()

    evaluator = Engine(validation_step)

    # Load the test data
    test_loader = Loader(test_x, test_y, batchsize=1024)
    
    print("Matrix Factorization test set loss:", evaluator.run(test_loader).output)
    
    log_dir = 'runs/ANN'
    writer = SummaryWriter(log_dir=log_dir)
    net = RecommenderNet(n_users=n_user, n_movies=n_item, writer=writer).to(device)
    net.load_state_dict(torch.load("net.pt"))
    
    batch_sz = 128
    for i in range(0, n_samples, batch_sz):
        limit =  min(i + batch_sz, n_samples)
        users_batch, movies_batch, rates_batch = users[i: limit], movies[i: limit], rates[i: limit]
        batches.append((torch.tensor(users_batch, dtype=torch.long), torch.tensor(movies_batch, dtype=torch.long),
                        torch.tensor(rates_batch, dtype=torch.float)))
    log_dir = 'runs/ANN_test'
    writer = SummaryWriter(log_dir=log_dir)
    
    for users_batch, movies_batch, rates_batch in batches:
        net.eval()
        with torch.no_grad():
            out = net(users_batch.to(device), movies_batch.to(device), [1, 5]).squeeze()
            loss = net.loss(rates_batch.to(device), out)
            optimizer.step()
            return loss.item()
        #print("Loss at epoch {} = {}".format(epoch, loss.item()))
    print("Test Loss = {}".format(loss.item()))
