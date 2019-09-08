import pandas as pd


data = pd.read_csv('/root/Downloads/kaggle_project/train.csv')
f1 = open('/root/Downloads/kaggle_project/train_names.txt','w')
f2 = open('/root/Downloads/kaggle_project/test_names.txt','w')
rows = data[data.pixels.notnull()]
im_labels = rows['Image'].tolist()
labels = [x[:-2] for x in im_labels]
labels = list(set(labels))
train_labels = labels[:5500]
test_labels = labels[5500:]
f1.write('\n'.join(train_labels))
f2.write('\n'.join(test_labels))
f1.close()
f2.close()
