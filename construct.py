from dataset import Dataset
from Network import MaskNet

batch_size = 5
data = Dataset('/root/Downloads/kaggle_project',batch_size,True)
model = MaskNet()
model.cuda()
epochs = 3
num = 0
for i in range(epochs):
    while num < data.number_train_images:
        images, train_data = data.get_next_batch()
        images['image'] = images['image'].cuda()
        model.train(images['image'], train_data)
        num += batch_size
    model.save()






