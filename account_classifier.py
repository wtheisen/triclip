# %%
import copy
import random

from imblearn.over_sampling import SMOTE

 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config as CFG

from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer

from imgcat import imgcat
 
from triclip import CLIPModel
from utils import make_train_valid_dfs, build_loaders
from collections import Counter


# %%
triclip = CLIPModel().to(CFG.device)
triclip.load_state_dict(torch.load('./test_model.pt'))
triclip.eval()

# %%
# Read data
train_df, valid_df, test_df = make_train_valid_dfs()
print(len(train_df))

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
train_loader = build_loaders(train_df, tokenizer, mode="train")
# valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

embedding_stances = pd.DataFrame(columns=[i for i in range(257)])


# %%
tqdm_object = tqdm(train_loader, total=len(train_loader))

for batch in tqdm_object:
    users = [path.split('/')[5] for path in batch["video_path"]]
    # print(Counter(users))
    # print('test')

    batch = {k: v.to(CFG.device) for k, v in batch.items() if k != 'video_path'}
    batch_embeddings = triclip.embed(batch)

    for i, user in enumerate(users):
        for e_i in range(3):
            embedding_stances = pd.concat([pd.DataFrame([[float(x) for x in batch_embeddings[e_i][i].cpu()] + [user]],
                                            columns=embedding_stances.columns),
                                            embedding_stances], 
                                            ignore_index=True)


# %%
X = embedding_stances.iloc[:, 0:256].values.tolist()
y = embedding_stances[256]

u_counts = Counter(y)
print(u_counts)

u = [[username] for username in y]
y = u
 
# Binary encoding of labels
encoder = LabelEncoder()
t = encoder.fit(y)
y_encoded = encoder.transform(y)


# Assuming y_encoded is your target variable after label encoding
# min_samples = y_encoded[y_encoded == np.argmin(np.bincount(y_encoded))].size
n_neighbors = min(5, max(1, u_counts.most_common()[-1][1] - 1))  # Adjust based on your minimum class size

oversample = SMOTE(k_neighbors=n_neighbors)
X, y_resampled = oversample.fit_resample(X, y_encoded)

# y = encoder.transform(y_resampled)
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y_resampled)
# print(ohe.categories_)

# y = ohe.transform(y_resampled)
# print(y)

# oversample = SMOTE()
# X, y = oversample.fit_resample(X, y)
# Convert to 2D PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
# print(len(X))
# print(len(y))

y = torch.tensor(y_resampled, dtype=torch.float32)#.reshape(-1, 1)
# print(y)

# %%
# Define two models
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(256, 768)
        self.relu = nn.ReLU()
        self.output = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
 

# %%
class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(256, 256)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 256)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

class DeepMulticlass(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(256, 256)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 256)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(256, 256)
        self.act4 = nn.ReLU()
        self.layer5 = nn.Linear(256, 256)
        self.act5 = nn.ReLU()
        self.output = nn.Linear(256, len(u_counts))
        # self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        x = self.act5(self.layer5(x))
        # x = self.sigmoid(self.output(x))
        # return x
        return torch.softmax(self.output(x), dim=1)

class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(256, 64)
        self.act = nn.ReLU()
        # self.output = nn.Linear(64, 33)
        self.output = nn.Linear(64, len(u_counts))
        
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        # return torch.softmax(x, dim=1).type(torch.float)
        return torch.softmax(x, dim=1)
        # return x
    
model = DeepMulticlass()
# %%
# Compare model sizes
model1 = Wide()
model2 = Deep()
# print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))  # 11161
# print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))  # 11041
 

# %%
# Helper function to train one model
# def model_train(model, X_train, y_train, X_val, y_val):
#     # loss function and optimizer
#     loss_fn = nn.CrossEntropyLoss()  # binary cross entropy
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
#     n_epochs = 300   # number of epochs to run
#     batch_size = 10  # size of each batch
#     batch_start = torch.arange(0, len(X_train), batch_size)
 
#     # Hold the best model
#     best_acc = - np.inf   # init to negative infinity
#     best_weights = None

#     for epoch in range(n_epochs):
#         model.train()
#         with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
#             bar.set_description(f"Epoch {epoch}")
#             for start in bar:
#                 # take a batch
#                 X_batch = X_train[start:start+batch_size]
#                 y_batch = y_train[start:start+batch_size]
#                 # forward pass
#                 y_pred = model(X_batch)
#                 loss = loss_fn(y_pred, y_batch)
#                 # backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 # update weights
#                 optimizer.step()
#                 # print progress
#                 acc = (y_pred.round() == y_batch).float().mean()
#                 bar.set_postfix(
#                     loss=float(loss),
#                     acc=float(acc)
#                 )
#         # evaluate accuracy at end of each epoch
#         model.eval()
#         y_pred = model(X_val)
#         acc = (y_pred.round() == y_val).float().mean()
#         acc = float(acc)
#         if acc > best_acc:
#             best_acc = acc
#             best_weights = copy.deepcopy(model.state_dict())
#     # restore model and return best accuracy
#     model.load_state_dict(best_weights)
#     return best_acc
 

# # %%
# # train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

# loss metric and optimizer
model = Multiclass()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# prepare model and training parameters
n_epochs = 5000
batch_size = 1000
# batches_per_epoch = len(X_train) // batch_size
batch_starts = torch.arange(0, len(X_train), batch_size)
print(len(X_train))
print(batch_starts)
 
best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []
 
# training loop
for epoch in tqdm(range(n_epochs), desc='Epochs'):
    epoch_loss = []
    epoch_acc = []
    # set model in training mode and run through each batch
    model.train()
    # with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        # bar.set_description(f"Epoch {epoch}")
    for i in batch_starts:
        # take a batch
        # start = i * batch_size
        # print(start)
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        # forward pass
        # print(X_batch)
        y_pred = model(X_batch)
        # print(y_pred)
        # print(y_pred.shape)
        # closest_indices = np.argmin(y_pred.detach.numpy(), axis=1)
        # print(closest_indices)
        # print(torch.argmin(y_pred, dim=1))
        # print(torch.argmax(y_pred, dim=1))
        # print(y_batch)
        # exit()
        # print(y_batch)
        # y_true = torch.argmax(y_batch, dim=1)
        # loss = loss_fn(torch.argmax(y_pred, dim=1).type(torch.float), y_batch)
        # encoded_arr = np.zeros((batch_size, len(u_counts) + 1), dtype=float)
        # print(encoded_arr)
        # for i in range(batch_size):
        #     encoded_arr[i][int(y_batch[i])] = 1.
        # encoded_arr[np.arange(batch_size), y_batch] = 1.
        loss = loss_fn(y_pred, y_batch.type(torch.long))
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        # compute and store metrics
        # print(torch.argmax(y_pred, 1))
        # print(torch.argmax(y_batch, 1))
        # print(torch.argmax(y_pred, 1) == y_batch)
        acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
        epoch_loss.append(float(loss))
        epoch_acc.append(float(acc))
        print(f'Loss: {loss}, Acc: {acc}')
        # bar.set_postfix(
        #     loss=float(loss),
        #     acc=float(acc)
        # )
    # set model in evaluation mode and run through the test set
    model.eval()
    y_pred = model(X_test)
    # print(y_pred)
    # print(y_test)
    # y_true = torch.argmax(y_test, dim=1)
    # ce = loss_fn(torch.argmax(y_pred, dim=1).type(torch.float), y_test)
    # ce = loss_fn(y_pred, y_true)
    ce = loss_fn(y_pred, y_test.type(torch.long))
    acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
    ce = float(ce)
    acc = float(acc)
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(ce)
    test_acc_hist.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")
 
# Restore best model
model.load_state_dict(best_weights)
 
# Plot the loss and accuracy
plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.show()
imgcat(plt.gcf())
plt.clf()
 
plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()
imgcat(plt.gcf())

# # %%
# # define 5-fold cross validation test harness
# kfold = StratifiedKFold(n_splits=5, shuffle=True)
# cv_scores_wide = []
# for train, test in kfold.split(X_train, y_train):
#     # create model, train, and get accuracy
#     model = Wide()
#     # train = train.tolist()
#     # test = test.tolist()
#     acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
#     print("Accuracy (wide): %.2f" % acc)
#     cv_scores_wide.append(acc)
# cv_scores_deep = []
# for train, test in kfold.split(X_train, y_train):
#     # create model, train, and get accuracy
#     model = Deep()
#     acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
#     print("Accuracy (deep): %.2f" % acc)
#     cv_scores_deep.append(acc)

# # %%
# # evaluate the model
# wide_acc = np.mean(cv_scores_wide)
# wide_std = np.std(cv_scores_wide)
# deep_acc = np.mean(cv_scores_deep)
# deep_std = np.std(cv_scores_deep)
# print("Wide: %.2f%% (+/- %.2f%%)" % (wide_acc*100, wide_std*100))
# print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))
 
# # %%
# # rebuild model with full set of training data
# if wide_acc > deep_acc:
#     print("Retrain a wide model")
#     model = Wide()
# else:
#     print("Retrain a deep model")
#     model = Deep()
# acc = model_train(model, X_train, y_train, X_test, y_test)
# print(f"Final model accuracy: {acc*100:.2f}%")
# torch.save(model.state_dict(), "accounts_best.pt")

# model.eval()
# with torch.no_grad():
#     # Test out inference with 5 samples
#     for i in range(5):
#         y_pred = model(X_test[i:i+1])
#         # print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")

#     # Plot the ROC curve
#     y_pred = model(X_test)
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#     plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
#     plt.title("Receiver Operating Characteristics")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     imgcat(plt.gcf())

