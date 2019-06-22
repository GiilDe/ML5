from get_prepared_data import get_prepared_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.optim as optim
from data_handling import X_Y_2_XY
import pandas as pd

train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()
train_X = torch.Tensor(train_X.to_numpy())
train_Y = torch.Tensor(train_Y.to_numpy())
train_Y = train_Y.type(dtype=torch.long)

# mlp = MLPClassifier(hidden_layer_sizes=[200, 74, 146, 180, 54], activation='tanh', verbose=True, max_iter=500, early_stopping=False)
# mlp.fit(train_X, train_Y)
# Y_hat = mlp.predict(validation_X)
# accuracy = accuracy_score(validation_Y, Y_hat)


class NN_classifier(torch.nn.Module):
    def __init__(self):
        super(NN_classifier, self).__init__()
        blocks = [nn.Linear(16, 200), nn.Dropout(0.16), nn.BatchNorm1d(200), nn.Linear(200, 82), nn.Dropout(0.16),
                  nn.BatchNorm1d(82), nn.Linear(82, 36), nn.Dropout(0.16), nn.BatchNorm1d(36), nn.Linear(36, 13)]
        self.classifier = nn.Sequential(*blocks)

    def forward(self, X):
        return self.classifier(X)

validation_X = torch.Tensor(validation_X.to_numpy())
validation_Y = torch.Tensor(validation_Y.to_numpy())
validation_Y = validation_Y.type(dtype=torch.long)

classifier = NN_classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                          shuffle=True, num_workers=2)

N = len(train_X)


for epoch in range(40):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(0, len(train_X), 5):
        # get the inputs; data is a list of [inputs, labels]

        inputs = train_X[i:i + 5, :]
        labels = train_Y[i:i + 5]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = classifier.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()


    print('[%d] loss: %.3f' %
          (epoch + 1, running_loss / 2000))
    running_loss = 0.0


    Y_hat = classifier.forward(validation_X)
    y_hat = torch.argmax(Y_hat, dim=1)

    accuracy = torch.sum(y_hat == validation_Y).item() / len(validation_X)
    print("validation accuracy: " + str(accuracy) + "\n")




