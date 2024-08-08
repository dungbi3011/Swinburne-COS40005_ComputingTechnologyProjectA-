from dataloader import train_loader  
from dataloader_for_testing import test_loader
from feature_extraction_model import Model
from sklearn import svm
from dataloader_for_testing import TEST_BATCH_SIZE
from dataloader import BATCH_SIZE

model = Model()

one_class_svm_model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

for batch_id, (images, labels) in enumerate(train_loader):
    if (images.size(0) >= BATCH_SIZE):
        if batch_id < 20:
            images = images.view(3, images.size(0), 224, 224)
            feature = model(images)
            feature = feature.detach().numpy()
            nsamples, depth, nx, ny = feature.shape
            feature = feature.reshape((nsamples, depth * nx * ny))
            one_class_svm_model.fit(feature)
            
for batch_id, (images, labels) in enumerate(test_loader):
    if (images.size(0) >= TEST_BATCH_SIZE):
        images = images.view(3, images.size(0), 224, 224)
        feature = model(images)
        feature = feature.detach().numpy()
        nsamples, depth, nx, ny = feature.shape
        feature = feature.reshape((nsamples, depth * nx * ny))
        predict_feature = one_class_svm_model.predict(feature)
        print(predict_feature)

