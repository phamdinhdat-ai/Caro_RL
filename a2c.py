import torch
import torch.nn as nn 
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F
import os 


class ActorCriticNet(nn.Module):
    def __init__(self, in_dim, out_dim, lr = 0.001, ckp_dir = 'checkoint/model'):
        super(ActorCriticNet, self).__init__()
        self.checkpoint = ckp_dir
        #extract feature 
        self.conv1 = nn.Conv2d(in_dim, 10, kernel_size= 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size= 2)
        self.conv3 = nn.Conv2d(20, 20, kernel_size= 2)
        self.conv4 = nn.Conv2d(20, 25, kernel_size= 2)
        # 720 = 6 * 6 * 20, 400 is a completely random number
        #create actor network and critic network
        #actor network(policy network)
        self.fc1 = nn.Linear(180, 256)
        self.policy = nn.Linear(256, out_dim)

        #critic network 
        self.fc2 = nn.Linear(180, 256)
        self.value = nn.Linear(256, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr= lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        #feature extracting
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 180)
        
        logits = F.relu(self.fc1(x))
        logits = self.policy(logits)

        value  = F.relu(self.fc2(x))
        value  = self.value(value)
        return logits, value[0]
    
    def save_checkpoint(self, file_name='actor_critic.pth'):
        model_folder_path = self.checkpoint
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    def load_checkpoint(self, file_name='actor_critic.pth'):

        file_name = os.path.join(self.checkpoint, file_name)
        self.load_state_dict(torch.load(file_name))

