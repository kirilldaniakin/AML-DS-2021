import numpy as np
from context import scripts
import scripts
import tensorflow as tf
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
    M = df_ratings.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    M = torch.from_numpy(M.mask(M>0, 1).to_numpy()).to(device)
    R = torch.from_numpy(R).to(device)
    
    n_user = R.shape[0]
    n_item = R.shape[1]

    mf = MF(R, n_user, n_item, writer=writer, k=k, c_vector=c_vector).to(device)
    mf.load_state_dict(torch.load("mf.pt"))
    # Setup TensorBoard logging
    log_dir = 'runs/simple_mf_01_' + str(datetime.now()).replace(' ', '_')
    writer = SummaryWriter(log_dir=log_dir)

    # Use Mean Squared Error as evaluation metric
    metrics = {'evaluation': MeanSquaredError()}

    # Create a supervised evaluator
    evaluator = create_supervised_evaluator(mf, metrics=metrics)
    test_loader = Loader(test_x, test_y, batchsize=1024)


    mf = MF(R, n_user, n_item, writer=writer, k=k, c_vector=c_vector).to(device)
    mf.load_state_dict(torch.load("mf.pt"))
    
    # Load the train and test data
    test_loader = Loader(test_x, test_y, batchsize=1024)

    def log_validation_results(engine):
        """
        Function to log the validation loss
        """
        # When triggered, run the validation set
        evaluator.run(test_loader)
        # Keep track of the evaluation metrics
        avg_loss = evaluator.state.metrics['evaluation']
        print("Epoch[{}] Validation MSE: {:.2f} ".format(engine.state.epoch, avg_loss))
        writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)


    evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation_results)
    evaluator.run(test_loader)
