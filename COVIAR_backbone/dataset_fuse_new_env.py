from torch.utils.data import Dataset
import torch
import torchvision 
from transforms import GroupRandomHorizontalFlip

class Synthetic_Dataset(Dataset):
    def __init__(self, datapath, tgtpath, train_len = 9535):
        self.datapath = datapath
        self.tgtpath = tgtpath
        self.train_len = train_len
        # self.args = args
        # self.transform = transform
        
    def __getitem__(self, index):
        input_i_mv_r = torch.load(self.datapath + 'video_' + str(index) + '.pth')  

        input_i = input_i_mv_r[0]
        input_mv = input_i_mv_r[1]
        input_r = input_i_mv_r[2]

        input_i = torch.squeeze(input_i)
        input_mv = torch.squeeze(input_mv)
        input_r = torch.squeeze(input_r)

        y = torch.load(self.tgtpath + 'label_'+ str(index) + '.pth')

        # if self.transform:
        # transform = torchvision.transforms.Compose(
        #             [
        #             GroupRandomHorizontalFlip(is_mv=True),
                    
        #             ])
        # input_mv = transform(input_mv) 
        # input_mv = torchvision.transforms.Compose(
        #             [
        #             GroupRandomHorizontalFlip(is_mv=True)
        #             ])(input_mv)

        return {'i': input_i, 'mv': input_mv, 'r': input_r, 'target': y}
    
    def __len__(self):
        if self.train_len == 9535:
            return 9535 
        elif self.train_len == 3783:
            return 3780
        else:
            return 5000
