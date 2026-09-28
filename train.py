import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset , DataLoader
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt 

#set random seeds for reproducibility 
torch.manual_seed(42)

df = pd.read_csv('fmnist_small.csv')
df.head()

#creating grid of 4x4 images 
fig, axes = plt.subplots(4, 4, figsize=(10,10))
fig.suptitle("First 16 images", fontsize = 16)

#plot the first 16 images from the dtatset
for i, ax in enumerate (axes.flat):
    img = df.iloc[i,1:]. values.reshape(28,28) #reshape to 28
    ax.imshow(img) #display in grayscale
    ax.axis('off') #remove axis for a cleaner look 
    ax.set_title(f"Label:{df.iloc[i,0]}") #show te label 
plt.tight_layout(rect = [0 , 0 , 1 , 0.96]) #adjust the layout to fit
plt.show()


#train test split
X = df.iloc[:,1:].values
y = df.iloc[:,0:].values

#scaling the feature 
X_train = X_train/225.0
X_test = X_test/255.0

#create CustomDataset Class
class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features = torch.tensor (features , dtype=torch.float32)
        self.labels = torch.tensor (labels , dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self , index):
        return self.features[index], self.labels[index]

#crete train_dataset object
train_dataset = CustomDataset(X_train , y_train)

#create test_dataset object
train_dataset = CustomDataset(X_test , y_test)

#create train and test loader
train_loader = DataLoader(train_dataset , batch_size=32, shuffle = True)
test_loader = DataLoader(test_dataset , batch_size=32 , shuffle = False)
#define a NN class

class MyNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()#invoke construc odf the parents clas
        self.model = nn.Sequential(
            nn.Linear(num_features,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward (self, x):
        return self.model(x)

epochs = 100
learning_rate = 0.1

#insittaite the model 
model = MyNN(X_train.shape[1])
#losso function
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


len(train_loader)

#traning loop 
for epochs in range (epochs):
    total_epoch_loss = 0

    for batch_features , batch_labels in train_loader:
        #forward pass 
        outputs = model(batch_features)
        #calculate loss
        loss = criterion(outputs , batch_labels)
        #backpass
        optimizer.zerograd()#gradient get cleared
        loss.backwards()

        #update grads
        optimizer.step()
        total_epoch_loss = total_epoch_loss + loss.item()

avg_loss = total_epoch_loss/len(train_loader)
print(f'Epoch: {epochs + 1} , Loss: {avg_loss}')

model.eval()
# evaluation code
total = 0
correct = 0

with torch.no_grad():

  for batch_features, batch_labels in test_loader:

    outputs = model(batch_features)

    _, predicted = torch.max(outputs, 1)

    total = total + batch_labels.shape[0]

    correct = correct + (predicted == batch_labels).sum().item()

print(correct/total)
