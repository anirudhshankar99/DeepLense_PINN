import torch
import numpy as np
import pandas as pd

class LensingDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, directory, transform=None, labels = [True, True, True]):
        """
        The dataset class

        :param csv_file: csv file which contains (image address, labels) pairs
        :param directory: Address of the csv file
        :param transform: Optional transform to be applied to the images before feeding into the model
        :param labels: Optional list that decides which classes are to be trained for
        """
        super(LensingDataset, self).__init__()
        self.index_file = pd.read_csv(directory+csv_file,header=None)
        self.u_length = len(self.index_file)//len(labels)
        for i in range(len(labels)):
            if labels[i] == False: 
                self.index_file = self.index_file.drop(list(range(self.u_length*i,self.u_length*(i+1))))
        self.directory = directory
        self.transform = transform
    
    def __len__(self):
        """
        Returns the length of the dataset, concatenated between all classes
        """
        return len(self.index_file)
    
    def __getitem__(self, index):
        """
        Returns the image, label pair
        
        :param index: Index of the image to be returned 
        """
        img_path = self.directory + self.index_file.iloc[index, 0]
        image = torch.tensor(np.load(img_path))
        label = int(self.index_file.iloc[index,1])
        y = [0 for _ in range(3)]
        y[label] = float(1)
        y = torch.tensor(y)
        if self.transform:
            image = self.transform(image)
        return (image,y)