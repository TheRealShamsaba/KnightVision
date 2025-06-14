import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class ChessNet(nn.Module):
    def __init__(self):
        super (ChessNet, self).__init__()
        # input shape: (batch, 12, 8 , 8)
        
        self.conv1 = nn.Conv2d(12, 64, kernel_size = 3 , padding = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3 , padding = 1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        
        # fully connected for move prediction (4096 classes)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 4096) # for 64 * 64 move classifiwe 
        
        # value head ( predict win/lose / draw )
        self.val_fc1 = nn.Linear(128 * 8 * 8, 256)
        self.val_fc2 = nn.Linear(256, 1)
        
    def forward(self , x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        flat = x.view(x.size(0), -1)
        
        # policy head (move probailities)
        policy = F.relu(self.fc1(flat))
        policy = self.fc2(policy)
        
        # value head (how goo is this board)
        value = F.relu(self.val_fc1(flat))
        value = torch.tanh(self.val_fc2(value))
        
        return policy, value
