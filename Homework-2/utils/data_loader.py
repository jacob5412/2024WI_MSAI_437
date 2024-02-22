from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, data):
        self.to_tensor = transforms.ToTensor()
        self.data = []

        for item in data:
            image = item["image"]
            image = self.to_tensor(image)
            self.data.append({"text": item["text"], "image": image})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomMTImageDataset(Dataset):
    def __init__(self, data):
        self.to_tensor = transforms.ToTensor()
        self.data = []

        for item in data:
            image = item["image"]
            image = self.to_tensor(image)
            self.data.append({"class": item["class"], "image": image})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
