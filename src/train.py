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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # get data and max name length
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
    df_ratings=df_ratings.replace({"movieId": map_m})
    df_ratings=df_ratings.replace({"userId": map_u})
    train_x = df_ratings[['userId', 'movieId']].to_numpy()
    train_y = df_ratings['rating'].to_numpy()
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
    print("MF train")
    n_user = R.shape[0]
    n_item = R.shape[1]

    # Define the Hyper-parameters
    lr = 1e-2  # Learning Rate
    k = 10  # Number of dimensions per user, item
    c_vector = 1e-6  # regularization constant

    # Setup TensorBoard logging
    log_dir = 'runs/simple_mf_01_' + str(datetime.now()).replace(' ', '_')
    writer = SummaryWriter(log_dir=log_dir)

    # Instantiate the MF class object
    model = MF(R, n_user, n_item, writer=writer, k=k, c_vector=c_vector).to(device)

    #print(model.user.weight)

    #for p in model.user.parameters():
    #    print(p)
    optimizer1 = torch.optim.SGD(model.user.parameters(), lr=lr)
    optimizer2 = torch.optim.SGD(model.item.parameters(), lr=lr)

    # Create a supervised trainer
    #trainer = create_supervised_trainer(model, optimizer, model.loss)

    #print(model.state_dict())

    #for p in model.parameters():
    #    print(p)

    #for p in model.parameters():
    #    p.requires_grad = False

    def train_step(engine, batch):
        model.train()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        x, y = batch[0].to(device), batch[1].to(device)
        #print('x,y:', x.size(), y)
        y_pred = model(x)
        loss = model.loss(x, y_pred, y)
        #print("Loss:", loss.item())
        #loss.backward()
        i = 0
        #print("Loss:" ,mae_loss_with_nans(y_pred, y))
        #print("item:", model.item.weight.size())
        #print("user:", model.user.weight.size()
        p = model.user.weight
        q = model.item.weight
        P = p.data
        p.grad = -(model.c_vector * p - ((p.mm(torch.transpose(q, 0, 1)) - R)*M).mm(q)) / (n_user*n_item)
        optimizer1.step() 
        q.grad = -(model.c_vector * q - torch.transpose((p.mm(torch.transpose(q, 0, 1)) - R)*M, 0, 1).mm(p)) / (n_user*n_item)
        optimizer2.step()
        return loss.item()

    trainer = Engine(train_step)

    # Load the train and test data
    train_loader = Loader(train_x, train_y, batchsize=4096)
    test_loader = Loader(test_x, test_y, batchsize=4096)


    def log_training_loss(engine, log_interval=93):
        """
        Function to log the training loss
        """
        model.itr = engine.state.iteration  # Keep track of iterations
        if model.itr % log_interval == 0:
            fmt = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
            # Keep track of epochs and outputs
            msg = fmt.format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output)
            print(msg)


    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=log_training_loss)

    # Run the model for 5 epochs
    trainer.run(train_loader, max_epochs=1)

    # Save the model to a separate folder
    torch.save(model.state_dict(), 'mf.pt')
    
    print("ANN train")
    #df_ratings, df_ratings_test = scripts.get_data(data_path="../data/cf")
    users = df_ratings['userId'].values
    movies = df_ratings['movieId'].values
    rates = df_ratings['rating'].values
    n_samples = len(rates)

    n_users, n_movies =  max(users)+1, max(movies)+1
    batches = []

    #Create batches
    batch_sz = 128
    for i in range(0, n_samples, batch_sz):
        limit =  min(i + batch_sz, n_samples)
        users_batch, movies_batch, rates_batch = users[i: limit], movies[i: limit], rates[i: limit]
        batches.append((torch.tensor(users_batch, dtype=torch.long), torch.tensor(movies_batch, dtype=torch.long),
                        torch.tensor(rates_batch, dtype=torch.float)))
    log_dir = 'runs/ANN'
    writer = SummaryWriter(log_dir=log_dir)
    net = RecommenderNet(n_users=n_user, n_movies=n_item, writer=writer).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=2)
    epochs = 5

    itr = 0
    for epoch in range(epochs):
        train_loss = 0
        for users_batch, movies_batch, rates_batch in batches:
            net.zero_grad()
            out = net(users_batch.to(device), movies_batch.to(device), [1, 5]).squeeze()
            loss = net.loss(rates_batch.to(device), out, itr)
            itr += 1
            loss.backward()
            optimizer.step()
            train_loss += loss
        scheduler.step(loss)
        print("Loss at epoch {} = {}".format(epoch, loss.item()))
    print("Last Loss = {}".format(loss.item()))
    
    torch.save(net.state_dict(), 'net.pt')
