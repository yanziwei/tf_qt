import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dataloader import common_dataloader


class TransformerModel(nn.Module):
    def __init__(self, feature_size=8, num_classes=3, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(feature_size, 256)
        self.fc2 = nn.Linear(256, num_classes * 2)

    def forward(self, src):
        transformer_output = self.transformer_encoder(src)
        x = F.relu(self.fc1(transformer_output[:, -1, :]))
        out = self.fc2(x).view(-1, 2, 3)
        return F.softmax(out, dim=-1)


def engine():
    num_epochs = 10
    learning_rate = 0.001
    device = "cuda"
    model = TransformerModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    trian_data = "train_data"
    test_data = "test_data"
    train_dataloader = common_dataloader(
        trian_data, sequence_length=2000, batch_size=64, num_samples=10000)
    test_dataloader = common_dataloader(
        test_data, sequence_length=2000, batch_size=1, num_samples=1000)

    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss1 = criterion(outputs[:, 0], labels[:, 0])
            loss2 = criterion(outputs[:, 1], labels[:, 1])
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(train_dataloader), loss.item()))

        model.eval()
        with torch.no_grad():
            correct = [0, 0]
            total = [0, 0]
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 2)
                for i in range(2):
                    total[i] += labels[:, i].size(0)
                    correct[i] += (predicted[:, i] ==
                                   labels[:, i]).sum().item()

            for i in range(2):
                print('Test Accuracy of the model on label {}: {:.4f} %'.format(
                    i, 100 * correct[i] / total[i]))


engine()
