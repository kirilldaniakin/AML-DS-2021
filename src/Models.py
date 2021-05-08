# Import PyTorch Packages
import torch
from torch import nn
import torch.nn.functional as F

class my_function(torch.autograd.Function):
    def forward(self, input, user_weight, item_weight, c_vector):
        self.saved_for_backward = [input, user_weight, item_weight, c_vector]
        # output = [do something with input and parameters]
        return output

    def backward(self, grad_output):
        input, parameters = self.saved_for_backward
        # grad_input = [derivate forward(input) wrt parameters] * grad_output
        #grad_input = 
        return grad_input

# Define the MF Model
class MF(nn.Module):
    # Iteration counter
    itr = 0

    def __init__(self, R, n_user, n_item, k=10, c_vector=1.0, writer=None):
        """
        :param n_user: User column
        :param n_item: Item column
        :param k: Dimensions constant
        :param c_vector: Regularization constant
        :param writer: Log results via TensorBoard
        """
        super(MF, self).__init__()

        # This will hold the logging
        self.writer = writer

        # These are the hyper-parameters
        self.k = k
        self.n_user = n_user
        self.n_item = n_item
        self.c_vector = c_vector
        self.R = R

        # The embedding matrices for user and item are learned and fit by PyTorch
        self.user = nn.Embedding(n_user, k)
        #self.user.weight.data.uniform_(-1, 1)
        self.item = nn.Embedding(n_item, k)
        #self.item.weight.data.uniform_(-1, 1)

    def __call__(self, train_x):
        """This is the most important function in this script"""
        # These are the user indices, and correspond to "u" variable
        user_id = train_x[:, 0]
        # These are the item indices, correspond to the "i" variable
        item_id = train_x[:, 1]
        # Initialize a vector user = p_u using the user indices
        try:
            vector_user = self.user(user_id)
        except:
            print(torch.max(user_id))    
        # Initialize a vector item = q_i using the item indices
        vector_item = self.item(item_id)
        
        # The user-item interaction: p_u * q_i is a dot product between the 2 vectors above
        ui_interaction = torch.sum(vector_user * vector_item, dim=1)
        return ui_interaction

    def loss(self, x, prediction, target, R=None):
        """
        Function to calculate the loss metric
        """
        # Calculate the Mean Squared Error between target = R_ui and prediction = p_u * q_i
        #assert not torch.isnan(target).any()
        #print(prediction)
        #print(target)
        if R is None:
            R = self.R
        loss_mse = 0    
        for i in range(x.size()[0]):
            #print(torch.index_select(torch.index_select(x,0,torch.tensor([i])), 1, torch.tensor([0])))
            #print(torch.index_select(torch.index_select(x,0,torch.tensor([i])), 1, torch.tensor([1])))
            print("i=", i)
            try:
                if torch.index_select(torch.index_select(R, 0, torch.tensor(torch.index_select(torch.index_select(x,0,torch.tensor([i])), 1, torch.tensor([0])).item())),1,torch.tensor(torch.index_select(torch.index_select(x,0,torch.tensor([i])), 1, torch.tensor([1])).item())).item()!=0:
                    loss_mse += F.mse_loss(prediction, target.squeeze())
                else:
                    loss_mse += torch.tensor(0)
            except:
                loss_mse += torch.tensor(0)
        loss_mse /= x.size()[0]    
        # Compute L2 regularization over user (P) and item (Q) matrices
        prior_user = l2_regularize(self.user.weight) * self.c_vector
        prior_item = l2_regularize(self.item.weight) * self.c_vector

        # Add up the MSE loss + user & item regularization
        total = loss_mse + prior_user + prior_item

        # This logs all local variables to tensorboard
        for name, var in locals().items():
            if type(var) is torch.Tensor and var.nelement() == 1 and self.writer is not None:
                self.writer.add_scalar(name, var, self.itr)
        if type(total) is int:        
            return torch.tensor(total)
        else: return total    

    #def backward(self, prediction, target):
        #output = my_function(input, self.parameters) # here you call the function!
        #return output    


def l2_regularize(array):
    """
    Function to do L2 regularization
    """
    loss = torch.sum(array ** 2.0)
    return loss
