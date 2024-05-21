# from libauc.datasets import CheXpert
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torchvision.transforms as tt
from torch.utils.data import Dataset, random_split, TensorDataset
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import os
import PIL.Image as Image
import torchvision

class CustomDataset(Dataset):
    def __init__(self, data, target, device=None, transform=None, isreg=False):
        self.transform = transform
        self.isreg = isreg
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
            if isreg:
                self.targets = target.float().to(device)
            else:
                self.targets = target.long().to(device)

        else:
            self.data = data.float()
            if isreg:
                self.targets = target.float()
            else:
                self.targets = target.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return (sample_data, label)  # .astype('float32')

class standard_scaling:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit_transform(self, data):
        self.std = np.std(data, axis=0)
        self.mean = np.mean(data, axis=0)
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data

    def transform(self, data):
        transformed_data = np.subtract(data, self.mean)
        transformed_data = np.divide(transformed_data, self.std)
        return transformed_data
    
# class TinyImageNetDataset(Dataset):
#     def __init__(self, root_dir, split='train', transform=None):
#         """
#         Args:
#             root_dir (string): Directory with all the images.
#             split (string): Either 'train', 'val', or 'test'.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.root_dir = root_dir
#         self.split = split
#         self.transform = transform
#         self.classes = os.listdir(os.path.join(root_dir, split))
#         self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
#         self.class_name = self._get_names()
#         self.images = self._load_images()

#     def _load_images(self):
#         images = []
#         for cls in self.classes:
#             cls_dir = os.path.join(self.root_dir, self.split, cls, 'images')
#             for image_file in os.listdir(cls_dir):
#                 image_path = os.path.join(cls_dir, image_file)
#                 images.append((image_path, self.class_to_idx[cls]))
#         return images

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path, label = self.images[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

#     def _get_names(self):
#         entity_dict = {}
#         # Open the text file
#         with open('tiny-imagenet-200/words.txt', 'r') as file:
#             # Read each line
#             for line in file:
#                 # Split the line into key and value using tab ('\t') as delimiter
#                 key, value = line.strip().split('\t')
        
#                 first = value.strip().split(',')
#                 # Add the key-value pair to the dictionary
#                 entity_dict[key] = first[0]
#                 # entity_dict.append(line)
#         return entity_dict

class TinyImageNet(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): Either 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root
#         self.root_dir = os.path.join(root, main_dir) 
        self.split = split
        self.transform = transform
        self.classes = []
        with open(os.path.join(self.root_dir, 'wnids.txt'), 'r') as f:
            self.classes = f.read().strip().split()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.class_name = self._get_names()
        self.images = self._load_images()        
        # self.repl_str = str_out

    def _load_images(self):
        images = []
        if self.split == 'train':
            for cls in self.classes:
                cls_dir = os.path.join(self.root_dir, self.split, cls, 'images')
                for image_file in os.listdir(cls_dir):
                    image_path = os.path.join(cls_dir, image_file)
                    images.append((image_path, self.class_to_idx[cls]))
                    
        elif self.split == 'val':
            val_dir = os.path.join(self.root_dir, self.split, 'images')
            image_to_cls = {}
            with open(os.path.join(self.root_dir, self.split, 'val_annotations.txt'), 'r') as f:
                for line in f.read().strip().split('\n'):
                    # print(line)
                    image_to_cls[line.split()[0].strip()] = line.split()[1].strip()                  
            for image_file in os.listdir(val_dir):
                # print(image_file)
                image_path = os.path.join(val_dir, image_file)
                images.append((image_path, self.class_to_idx[image_to_cls[image_file]]))
                
            # for cls in self.classes:
            #     cls_dir = os.path.join(self.root_dir, self.split, cls, 'images')
            #     for image_file in os.listdir(cls_dir):
            #         image_path = os.path.join(cls_dir, image_file)
            #         images.append((image_path, self.class_to_idx[cls]))
            
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def _get_names(self):
        entity_dict = {}
        # Open the text file
        with open(os.path.join(self.root_dir, 'words.txt'), 'r') as file:
            # Read each line
            for line in file:
                # Split the line into key and value using tab ('\t') as delimiter
                key, value = line.strip().split('\t')
        
                first = value.strip().split(',')
                # Add the key-value pair to the dictionary
                entity_dict[key] = first[0]
                # entity_dict.append(line)
        return entity_dict

def loader(dataset, dirs="./cifar10", trn_batch_size=64, val_batch_size=64, tst_batch_size=1000):
    

    if dataset.lower() == "imagenet":
        # Define the data transforms
       
        traindir = os.path.join(dirs, 'train')
        valdir = os.path.join(dirs, 'val')
        

        fullset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        )

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        
    elif dataset.lower() == "tinyimagenet":
        transform_default = transforms.Compose([
        transforms.Resize(256), ##before was 150
        transforms.CenterCrop(224), ##added crop 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
        ])

#         fulldataset = TinyImageNetDataset(root_dir=dirs, transform=transform_default)
#         fullset, testset, _ = random_split(fulldataset, [80000, 10000, 10000])
        fullset = TinyImageNet(dirs, 'train', transform=transform_default)
        testset = TinyImageNet(dirs, 'val', transform=transform_default)
        
    if dataset.lower() == "cifar10":
        train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        valid_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
           
        
        fullset = datasets.CIFAR10(root='./data10', train=True, download=True, transform=train_tfms)
        testset = datasets.CIFAR10(root='./data10', train=False, download=True, transform=valid_tfms)

    
    
    elif dataset.lower() == "fashionmnist":
        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])
        fullset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
        testset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
    

    elif dataset.lower() == "boston":
        num_cls = 1
        seed = 42
        x_trn, y_trn = load_boston(return_X_y=True)
        # create train and test indices
        #train, test = train_test_split(list(range(X.shape[0])), test_size=.3)
        x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn, y_trn, test_size=0.2, random_state=seed)
        x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, test_size=0.1, random_state=seed)
        scaler = standard_scaling()
        x_trn = scaler.fit_transform(x_trn)
        x_val = scaler.transform(x_val)
        x_tst = scaler.transform(x_tst)
        y_trn = y_trn.reshape((-1, 1))
        y_val = y_val.reshape((-1, 1))
        y_tst = y_tst.reshape((-1, 1))
#         if isnumpy:
#             fullset = (x_trn, y_trn)
#             valset = (x_val, y_val)
#             testset = (x_tst, y_tst)
#         else:
        fullset = CustomDataset(torch.from_numpy(x_trn), torch.from_numpy(y_trn), isreg=True)
#         valset = CustomDataset(torch.from_numpy(x_val), torch.from_numpy(y_val), isreg=True)
        testset = CustomDataset(torch.from_numpy(x_tst), torch.from_numpy(y_tst), isreg=True)

    elif dataset.lower() == "cifar100":
            
        train_data = datasets.CIFAR100(root='./data100', train=True, download=True)
        x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

        mean = np.mean(x, axis=(0, 1))/255
        std = np.std(x, axis=(0, 1))/255
        mean=mean.tolist()
        std=std.tolist()
            
        transform_train = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(mean,std,inplace=True)])
        transform_test = tt.Compose([tt.ToTensor(), tt.Normalize(mean,std)])

            

        fullset = datasets.CIFAR100("./data100", train=True, download=True, transform=transform_train) 
        testset = datasets.CIFAR100("./data100", train=False, download=True, transform=transform_test)
            
#     seed = 42
#     validation_set_fraction = 0.1
#     num_fulltrn = len(fullset)
#     num_val = int(num_fulltrn * validation_set_fraction)
#     num_trn = num_fulltrn - num_val

#     trainset, valset = random_split(fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed))

    
    # Creating the Data Loaders
    trainloader = torch.utils.data.DataLoader(fullset, batch_size=trn_batch_size,
                                              shuffle=False, pin_memory=True, num_workers=1)

    valloader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=1)

#     testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
#                                               shuffle=False, pin_memory=True, num_workers=1)
    
    return trainloader, valloader, fullset, testset