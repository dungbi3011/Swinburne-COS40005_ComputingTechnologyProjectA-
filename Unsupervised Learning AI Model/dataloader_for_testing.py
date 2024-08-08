import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import Dataset

test_data_directory = 'extracted_frames_for_testing'

TEST_BATCH_SIZE = 32

transform_train = transforms.Compose([transforms.Resize((224, 224)), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_dataset = Dataset(
    root=test_data_directory, 
    transform=transform_train,
    target_transform=None)

test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)


