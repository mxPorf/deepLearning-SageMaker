#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import json
import os

#Import Sagemaker debug tool
import smdebug.pytorch as smd


def test(model, test_loader, loss_criterion, hook, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss = loss_criterion(output, target).item()
    test_loss /= len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, loss_criterion, optimizer, epoch, hook, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    running_loss=0
    correct=0
    for data, target in train_loader:
        data=data.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        #NOTE: Notice how we are not changing the data shape here
        # This is because CNNs expects a 3 dimensional input
        pred = model(data)
        loss = loss_criterion(pred, target)
        running_loss+=loss
        loss.backward()
        optimizer.step()
        pred=pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Train Epoch:{epoch} \t Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")
    
def net():
    '''
    Initializes a pretrained CNN model for image classification
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    
    #In the case of the Stanford Cars dataset, there are 196 different types of cars
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 196))
    
    return model

def _create_train_loader(path, batch_size):
    dataset = datasets.StanfordCars(
        path,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Resize(3,244,244)]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
def _create_test_loader(path, batch_size):
    dataset = datasets.StanfordCars(
        path,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Resize(3,244,244)]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

def main(args):
    #Use gpu, if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    '''
    Initialize a model by calling the net function
    '''
    model=net()
    model=model.to(device)
    '''
    Create loss cost function and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    '''
    Train the model
    '''
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
     
    train_loader = _create_train_loader(args.data_dir, args.batch_size)
    test_loader = _create_test_loader(args.data_dir, 512)
    
    for epoch in range(1, args.epochs+1):
        model=train(model, train_loader, loss_criterion, optimizer, epoch, hook, device)
        test(model, test_loader, loss_criterion, hook, device)
    '''
    Save the trained model
    '''
    torch.save(model.state_dict(), args.output_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="ResNet50 image classifier, with sagemaker profiler and debugger")
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=.001,
        metavar="N",
        help="input learning rate for training (default: 0.001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=32,
        metavar="N",
        help="input number of epochs in the training (default: 32)",
    )
    parser.add_argument(
        "--output-path",
        default='./saved_model',
        help="input path to store the trained model (default: ./saved_model)",
    )
    
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    
    args=parser.parse_args()
    
    main(args)
