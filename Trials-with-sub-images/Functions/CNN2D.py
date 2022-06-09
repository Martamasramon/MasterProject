import numpy as np
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt

class CNN2D(nn.Module):
    def __init__(self,img_width,pca_components,dropout_rate):
        super(CNN2D, self).__init__()
        SIZE1 = 10
        SIZE2 = 10
        SIZE3 = int(SIZE2 * img_width/2 * img_width/2)
        SIZE4 = 100

        self.conv1 = torch.nn.Conv2d(pca_components,SIZE1,kernel_size=1,padding='same')    #[SIZE1, IMG_WIDTH, IMG_WIDTH]
        self.relu1 = torch.nn.ReLU()                                                       #[SIZE1, IMG_WIDTH, IMG_WIDTH]
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, padding=0)                            #[SIZE1, IMG_WIDTH/2, IMG_WIDTH/2]
        self.conv2 = torch.nn.Conv2d(SIZE1,SIZE2,kernel_size=1,padding='same')             #[SIZE2, IMG_WIDTH/2, IMG_WIDTH/2]
        self.relu2 = torch.nn.ReLU()                                                       #[SIZE2, IMG_WIDTH/2, IMG_WIDTH/2]

        self.flat = torch.nn.Flatten(1)                                                    #[SIZE2 * IMG_WIDTH/2 * IMG_WIDTH/2]
        self.drop = torch.nn.Dropout(p=dropout_rate)                                       #[SIZE2 * IMG_WIDTH/2 * IMG_WIDTH/2]

        self.fc1 = torch.nn.Linear(SIZE3, SIZE4)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(SIZE4, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.relu2(output)

        output = self.flat(output)
        output = self.drop(output)

        output = self.fc1(output)
        output = self.relu3(output)
        output = self.fc2(output)
        output = self.softmax(output)
        return output

def train_model(model, data, target, learning_rate, num_epochs, img_width, pca_components):
    # Reshape data
    X = np.reshape(data,(-1,pca_components,img_width,img_width))
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(target).long()

    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None,
            sampler=torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(dataset), batch_size=100000, drop_last=False))

    # Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train();

    loss_history = []
    for _ in tqdm.trange(num_epochs):
            for _, (inputs, targets) in enumerate(data_loader):
                    optimizer.zero_grad()
                    y_pred = model(inputs)
                    loss = criterion(y_pred, targets)
                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item())

    return model

def test_model(model, data, img_width, pca_components):
    X = np.reshape(data,(-1,pca_components,img_width,img_width))
    tensor_X = torch.Tensor(X)

    # Evaluate model
    model.eval();
    y_pred = model(tensor_X)

    return y_pred

def get_error(results, target):
    # Analyse performance of model for training data
    result = []
    error = []

    for i in range(len(results)):
        array = results[i].detach().numpy()
        index = np.argmax(array)

        result.append(index*0.5)
        error.append(abs(target[i] - index))
    
    total_error = round(np.mean(error), 4)
    return total_error, result

def analyse_train_test(train_results, test_results, target, cutoff, parameters, plot=0, text=0):
    train_error, result_train = get_error(train_results, target[:cutoff])
    test_error,  result_test  = get_error(test_results,  target[cutoff:])

    if plot == 1:
        title = ','.join('='.join((key,str(val))) for (key,val) in parameters.items())
        fig, axs = plt.subplots(1,2, figsize=(20,5))
        axs[0].plot(target[:cutoff])
        axs[0].plot(result_train)
        axs[1].plot(target[cutoff:])
        axs[1].plot(result_test)
        fig.suptitle(title)

        plt.savefig('network_results.png')

    if text == 1:
        temp = ''.join(['_' if x==0 else '^' for x in result_test])
        written_plot = '\n'.join([temp[i:i+75] for i in range(0, len(temp), 75)])
        
        with open('network_results.txt', 'a') as f:
            f.write('---------------------------------- \n')
            f.write(f"{'Parameter':<20}{'Value':^15}" + '\n')
            f.write('---------------------------------- \n')
            for key in parameters:
                f.write(f"{key:<20}{str(parameters[key]):^15}" + '\n')
            f.write('---------------------------------- \n')
            f.write('Training error = ' + str(train_error*100) + '%\n')
            f.write('Test error = ' + str(test_error*100) + '%\n')
            f.write(written_plot)
            f.write('\n---------------------------------- \n\n\n\n')
            
    return result_train, result_test, test_error
