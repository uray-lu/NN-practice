#%%
import pandas as pd
import torch
from model import network
from matplotlib import pyplot as plt
import time 

#%%
# load data
training_data = pd.read_csv('/Users/ray/Desktop/educations/nn-modual-checkpoint/dataset/creditcard_train.csv', index_col=0)
train_x_origin, train_y_origin = training_data[training_data.columns[:30]], training_data[training_data.columns[-1]]

print(f'x_shape: {train_x_origin.shape}')
print(f'y_shape: {train_y_origin.shape}')


#transfer the data into array
train_x_origin_array = train_x_origin.values
train_y_origin_array = train_y_origin.values


# create tensor
x = torch.tensor(train_x_origin_array, dtype=torch.float32)
y = torch.tensor(train_y_origin_array, dtype=torch.float32).unsqueeze(1)
print(f'x_shape: {x.shape}')
print(f'y_shape: {y.shape}')

test_data = pd.read_csv('/Users/ray/Desktop/educations/nn-modual-checkpoint/dataset/creditcard_test.csv', index_col=0)
test_feature = test_data[test_data.columns[:30]].values
test_label = test_data[test_data.columns[-1]].values

test_feature = torch.tensor(test_feature, dtype=torch.float32)
test_label = torch.tensor(test_label, dtype =torch.float32).unsqueeze(1)


def training(Model, Feature, Labels, LossFunction, LearningRate, Optimizer, Device, Remark:str = 'No comment' , Epoch:int = 1000):

    if Device == 'cpu':
        device = torch.device('cpu')
        print('------ Training With CPU ------')
    elif Device == 'mps':
        device = torch.device('mps')
        print('------ Training With GPU ------')
    else:
        print('------ Chose a device first!! ------')
    
    model = Model.to(device)
    feature = Feature.to(device)
    labels = Labels.to(device)
    LossFunction = LossFunction.to(device)

    if Optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate, momentum=0.9)

    LossFunction = LossFunction.to(device)

    #Training
    
    loss_record = []
    start_time = time.time()
    model.train()
    for _ in range(Epoch):
        optimizer.zero_grad()
        output = model(feature)
        loss = LossFunction(output, labels)
        
        loss.backward()
        optimizer.step()
        

        # Record the model loss
        loss_record.append(loss.item())
        # Record the model weights
        if _ % 100 == 0 :
            print(f'Epoch : {_} | loss: {loss.item()}')
        
    end_time = time.time()
    training_time = end_time - start_time
    training_min = int(training_time)// 60
    training_second = training_time % 60
    print(f'------Training time: {training_min} minute{training_second:.2f} seconds-----')
    
    #save model weight
    torch.save(model.state_dict(), f"model_weight_{Optimizer}_{Remark}.pth")

    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(loss_record)
    plt.title('Loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    return model

#%%
# Dealing with data imbalance
pos_weight = torch.tensor([1.5], dtype= torch.float32).to(device = 'cpu')
trained_network = training(network(), x, y, torch.nn.BCEWithLogitsLoss(weight=pos_weight), 0.0001, Optimizer='Adam', Device = 'cpu',Remark='test_fro_6000_epoch', Epoch = 6000)

#%%
#Evaluate the Model
trained_network.eval()
with torch.no_grad():

    threshold = 0.5

    pred = trained_network(test_feature)
    pred = torch.sigmoid(pred)
    pred = (pred >= threshold).squeeze().numpy()
    print("pred values:", pred)

    test_accuracy = (pred == test_label.squeeze().numpy()).mean()
    formatted_percentage = "{:.2f}%".format(test_accuracy*100)
    print(f"Test Accuracy: {formatted_percentage}")




# %%
