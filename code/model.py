from torch import nn
import torch
from tqdm import tqdm

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, output_dim, device='cpu'):
        super(SiameseNetwork, self).__init__()
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        # self.gru = nn.GRU(input_size=input_dim, hidden_size=feature_dim)

        self.fc11 = nn.Linear(input_dim, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, feature_dim)

        self.fc21 = nn.Linear(feature_dim*2, 128)
        self.fc22 = nn.Linear(128, 128)
        self.fc23 = nn.Linear(128, output_dim)

        self.device = device

    def forward_Siamese(self, x):
        # x = self.gru(x)

        x = self.fc11(x)
        x = self.activation(x)
        x = self.fc12(x)
        x = self.activation(x)
        x = self.fc13(x)
        # x = self.softmax(x)
        return x

    def forward_feature(self, x):
        x = self.fc21(x)
        x = self.activation(x)
        x = self.fc22(x)
        x = self.activation(x)
        x = self.fc23(x)
        x = self.softmax(x)
        return x

    def forward(self, input):
        features = None
        for sample1, sample2 in input:
            # sample1 = torch.reshape(torch.FloatTensor(sample1).to(self.device), (-1, 1, 1))
            # sample2 = torch.reshape(torch.FloatTensor(sample2).to(self.device), (-1, 1, 1))
            sample1 = torch.reshape(torch.FloatTensor(sample1).to(self.device), (1, -1))
            sample2 = torch.reshape(torch.FloatTensor(sample2).to(self.device), (1, -1))
            feature1 = self.forward_Siamese(sample1)
            feature2 = self.forward_Siamese(sample2)
            # feature = torch.cat((feature1[-1,0,:], feature2[-1,0,:]), 0)
            # feature = torch.reshape(feature, (1, -1))
            feature = torch.cat((feature1, feature2), 1)
            if features is None:
                features = feature
            else:
                features = torch.cat((features, feature), 0)
        x = self.forward_feature(features)
        return x

    def forward_single(self, sample1, sample2):
        # sample1 = torch.reshape(torch.FloatTensor(sample1).to(self.device), (-1, 1, 1))
        # sample2 = torch.reshape(torch.FloatTensor(sample2).to(self.device), (-1, 1, 1))
        sample1 = torch.reshape(torch.FloatTensor(sample1).to(self.device), (1, -1))
        sample2 = torch.reshape(torch.FloatTensor(sample2).to(self.device), (1, -1))
        # feature1, _ = self.forward_Siamese(sample1)
        # feature2, _ = self.forward_Siamese(sample2)
        feature1 = self.forward_Siamese(sample1)
        feature2 = self.forward_Siamese(sample2)
        # feature = torch.cat((feature1[-1,0,:], feature2[-1,0,:]), 0)
        # feature = torch.reshape(feature, (1, -1))
        feature = torch.cat((feature1, feature2), 1)
        x = self.forward_feature(feature)
        return x


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(MLPNetwork, self).__init__()
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(input_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)

        self.device = device

    def forward_feature(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def forward(self, input):
        features = None
        for sample1, sample2 in input:
            sample1 = torch.reshape(torch.FloatTensor(sample1).to(self.device), (1, -1))
            sample2 = torch.reshape(torch.FloatTensor(sample2).to(self.device), (1, -1))
            feature = torch.cat((sample1, sample2), 1)
            if features is None:
                features = feature
            else:
                features = torch.cat((features, feature), 0)
        x = self.forward_feature(features)
        return x

    def forward_single(self, sample1, sample2):
        sample1 = torch.reshape(torch.FloatTensor(sample1).to(self.device), (1, -1))
        sample2 = torch.reshape(torch.FloatTensor(sample2).to(self.device), (1, -1))
        feature = torch.cat((sample1, sample2), 1)
        x = self.forward_feature(feature)
        return x
