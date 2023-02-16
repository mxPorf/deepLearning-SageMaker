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

from torchvision.datasets.stanford_cars import StanfordCars


#Import Sagemaker debug tool
try:
    import smdebug.pytorch as smd
except Exception as e:
    print(e)
    

def test(model, test_loader, loss_criterion, hook, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    if hook: 
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
    if hook:
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
    
def net(num_output_classes):
    '''
    Initializes a pretrained CNN model for image classification
    '''
    model = models.resnet50(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    
    #In the case of the Stanford Cars dataset, there are 196 different types of cars
    model.fc = nn.Sequential(
                   nn.Linear(num_features, num_output_classes))
    
    return model

def _create_train_loader(path, batch_size, download):
    dataset = StanfordCars(
        path,
        split='train',
        download=download,
        transform=transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Resize([244,244])]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
def _create_test_loader(path, batch_size, download):
    dataset = StanfordCars(
        path,
        split='test',
        download=download,
        transform=transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Resize([244,244])]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

def model_fn(model_dir):
    model = net(196)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model

def main(args):
    #Use gpu, if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    '''
    Initialize a model by calling the net function
    '''
    model=net(args.output_classes)
    model=model.to(device)
    '''
    Create loss cost function and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    '''
    Train the model
    '''
    try:
        hook = smd.Hook.create_from_json_file()
        # hook.register_module(model)
        hook.register_hook(model)
        hook.register_loss(loss_criterion)
    except Exception:
        hook = None
     
    train_loader = _create_train_loader(args.data_dir, args.batch_size, args.download)
    test_loader = _create_test_loader(args.data_dir, 512, args.download)
    
    for epoch in range(1, args.epochs+1):
        train(model, train_loader, loss_criterion, optimizer, epoch, hook, device)
        test(model, test_loader, loss_criterion, hook, device)
    '''
    Save the trained model
    '''
    full_path=os.path.join(args.model_dir, "model.pth")
    torch.save(model.to('cpu').state_dict(), full_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="ResNet50 image classifier, with sagemaker profiler and debugger")
    parser.add_argument(
        "--batch-size",
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
        "--output-classes",
        default=196,
        type=int,
        help="input number of prossible classes for prediction (default: 196)",
    )
    parser.add_argument(
        "--download",
        type=int,
        default=1,
        metavar="N",
        help="input whether to download or not download the datasets used for training (default: 1)",
    )
    parser.add_argument(
        "--out-dir",
        default='/opt/ml/profile/',
        help="input path where the profiler will write the reports (default: /opt/ml/profile/)",
    )
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS", '{"SM_HOSTS": ""}')))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST",''))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR","trained_model.pth"))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING",'./'))
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS",0))
    
    
    args=parser.parse_args()
    
    main(args)
