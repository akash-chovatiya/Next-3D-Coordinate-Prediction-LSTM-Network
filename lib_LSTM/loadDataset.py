# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:55:52 2022

@author: Chovatiya
"""

from torch.utils.data import Dataset

class augmented_coordinates(Dataset):
    def __init__(self, loc_3d, percentage):
        super().__init__()
        self.loc_3d = loc_3d
        self.percent = percentage
        
    def __getitem__(self, idx):
        loc_3d = self.loc_3d * ((100-(idx*self.percent))/100)
        sample = loc_3d[:-1]
        target = loc_3d[1:]
        
        return sample, target
    
    def __len__(self):
        return len(self.folder)