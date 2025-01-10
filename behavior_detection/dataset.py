from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import cv2
import numpy as np
from torchvision import transforms
import json


block_size = 16
downsampling_rate = 3

activities = ['Feeding', 'Preening', 'Swimming', 'Walking', 'Alert', 'Flying', 'Resting']

id_to_label = {i: activities[i] for i in range(len(activities))}
label_to_id = {v: k for k, v in id_to_label.items()}

distribution_path = 'annotations/splits.json'

def get_distribution():
    with open(distribution_path) as f:
        data = json.load(f)
    return data

class Dataset(Dataset):
    def __init__(self, frame_paths, labels):
        self.frame_paths = frame_paths
        self.labels = labels

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def __len__(self):
        return len(self.frame_paths)        
    
    def __getitem__(self, idx):

        # Load the images from the frame paths
        frames = []
        for frame_path in self.frame_paths[idx]:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frames = np.array(frames)
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)

        frames = frames.float() / 255.0  # Normalize pixel values to [0, 1]
        frames = self.normalize(frames)

        frames = frames.permute(1,0,2,3)

        frames = frames.to(self.device)
        
        # Convert labels to one-hot encoding
        label = torch.tensor(self.labels[idx]).to(self.device)
        #one_hot_label = torch.zeros(8)
        #one_hot_label[label] = 1
        #one_hot_label = one_hot_label.to(self.device)
        
        return frames, label

def process_CSV(csv_path, data_distribution, statistics=False):
    
    df = pd.read_csv(csv_path, delimiter=';')
    dataset = {'train_set': {'frames': [], 'labels': []}, 'val_set': {'frames': [], 'labels': []}, 'test_set': {'frames': [], 'labels': []}}

    discarded = 0

    actions = np.array([])
    # Dictionary to store the length of each action in np arrays
    action_lengths = {}
    for i in range(len(activities)):
        action_lengths[i] = np.array([])

    for index, row in df.iterrows():
        video_name = row['video_name']
        bird_id = row['bird_id']
        action_id = int(row['action_id'])
        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])

        action_frames = []

        # Downsample the dataset, to keep only one per 3 frames
        for i in range(start_frame, end_frame, downsampling_rate):
            action_frames.append(f'crops/{video_name}/{bird_id}/frame_{str(i).zfill(5)}.jpg')
        
        if len(action_frames) <= block_size:
            discarded += 1
            continue
        elif len(action_frames) >= 2 * block_size:
            # Divide the action in two or more blocks
            for i in range(0, len(action_frames), block_size):

                if i + block_size > len(action_frames):
                    break

                if video_name in data_distribution['train_set']:
                    dataset['train_set']['frames'].append(action_frames[i:i+block_size])
                    dataset['train_set']['labels'].append(action_id)
                elif video_name in data_distribution['val_set']:
                    dataset['val_set']['frames'].append(action_frames[i:i+block_size])
                    dataset['val_set']['labels'].append(action_id)
                elif video_name in data_distribution['test_set']:
                    dataset['test_set']['frames'].append(action_frames[i:i+block_size])
                    dataset['test_set']['labels'].append(action_id)

                if statistics:
                    actions = np.append(actions, action_id)
                    action_lengths[action_id] = np.append(action_lengths[action_id], np.array([block_size]))

        elif len(action_frames) > block_size:
            action_frames = action_frames[:block_size]

            if statistics:
                actions = np.append(actions, action_id)
                action_lengths[action_id] = np.append(action_lengths[action_id], np.array([block_size]))

            if video_name in data_distribution['train_set']:
                dataset['train_set']['frames'].append(action_frames)
                dataset['train_set']['labels'].append(action_id)
            elif video_name in data_distribution['val_set']:
                dataset['val_set']['frames'].append(action_frames)
                dataset['val_set']['labels'].append(action_id)
            elif video_name in data_distribution['test_set']:
                dataset['test_set']['frames'].append(action_frames)
                dataset['test_set']['labels'].append(action_id)

    if statistics:
        # Save numpy arrays to disk
        np.save('annotations/stats/activities.npy', actions)
        for key, value in action_lengths.items():
            np.save(f'annotations/stats/activity_{key}_lengths.npy', value)

    print('Included in dataset: train: ' + str(len(dataset['train_set']['frames'])) + ' val: ' + str(len(dataset['val_set']['frames'])) + ' test: ' + str(len(dataset['test_set']['frames'])))
    print('Samples discarted for dataset: ' + str(discarded))
    

    return dataset

def get_dataloader(batch_size=32, shuffle=True, statistics = False, csv_path = '/dataset/annotations/crops.csv'):

    data_distribution = get_distribution()

    data = process_CSV(csv_path, data_distribution, statistics)
    
    train_dataset = Dataset(data['train_set']['frames'], data['train_set']['labels'])
    val_dataset = Dataset(data['val_set']['frames'], data['val_set']['labels'])
    test_dataset = Dataset(data['test_set']['frames'], data['test_set']['labels'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader