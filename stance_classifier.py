# %%
import copy
import random

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import stance_config as CFG

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer

from imgcat import imgcat
from stances import stance_map
 
from triclip import CLIPModel
from utils import make_train_valid_dfs, build_loaders


# %%

models = {
    'triCLIP-100': './100t_50e_best.pt',
    'triCLIP-1k': './1000t_50e_best.pt',
    'triCLIP-10k': './10000t_50e_best.pt',
    'triCLIP-50k': './50000t_50e_best.pt'
}

model_stats = {}
# %%
# Read data
train_df, valid_df, test_df = make_train_valid_dfs()
users = [path.split('/')[7] for path in train_df["video_path"]]
print(len(train_df))
print(set(users))

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
train_loader = build_loaders(train_df, tokenizer, mode="train")
# valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

for model_name, model_path in tqdm(models.items(), desc='Models'):
    triclip = CLIPModel().to(CFG.device)
    triclip.load_state_dict(torch.load(model_path))
    triclip.eval()
    tqdm.write('Model loaded!')

    embedding_stances = pd.DataFrame(columns=[i for i in range(257)])

    # %%
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        users = [path.split('/')[7] for path in batch["video_path"]]

        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != 'video_path'}
        batch_embeddings = triclip.embed(batch)

        for i, user in enumerate(users):
            for e_i in range(3):
                try:
                    embedding_stances = pd.concat([pd.DataFrame([[float(x) for x in batch_embeddings[e_i][i].cpu()] + [stance_map[user]]],
                                                    columns=embedding_stances.columns),
                                                    embedding_stances], 
                                                    ignore_index=True)
                except:
                    pass



    # %%
    X = embedding_stances.iloc[:, 0:256].values.tolist()
    y = embedding_stances[256]
    
    # Binary encoding of labels
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    # Convert to 2D PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    # print(len(X))
    # print(len(y))

    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

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
    

    # %%
    # Compare model sizes
    model1 = Wide()
    model2 = Deep()
    # print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))  # 11161
    # print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))  # 11041
    

    # %%
    # Helper function to train one model
    def model_train(model, X_train, y_train, X_val, y_val):
        # loss function and optimizer
        loss_fn = nn.BCELoss()  # binary cross entropy
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
        n_epochs = 300   # number of epochs to run
        batch_size = 256  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)
    
        # Hold the best model
        best_acc = - np.inf   # init to negative infinity
        best_weights = None

        for epoch in range(n_epochs):
            model.train()
            with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    acc = (y_pred.round() == y_batch).float().mean()
                    bar.set_postfix(
                        loss=float(loss),
                        acc=float(acc)
                    )
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
            acc = float(acc)
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        return best_acc
    

    # %%
    trials = 10
    accs = []
    rocs = []

    for i in range(trials):
        # train-test split: Hold out the test set for final model evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

        # %%
        # define 5-fold cross validation test harness
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
        
        # %%
        # rebuild model with full set of training data
        # if wide_acc > deep_acc:
        #     print("Retrain a wide model")
        #     model = Wide()
        # else:
        model = Deep()

        acc = model_train(model, X_train, y_train, X_test, y_test)
        # print(f"Model accuracy: {acc*100:.2f}%")
        accs.append(acc)
        # torch.save(model.state_dict(), "stances_best.pt")

    model.eval()
    with torch.no_grad():
        # Test out inference with 50 samples
        # print('cope')
        # for i in range(len(y_test)):
        #     print('blah')
        #     y_pred = model(X_test[i:i+1])

        # Plot the ROC curve
        y_pred = model(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        model_stats[model_name] = (fpr, tpr)

    print(f'{model_name} accuracy over 5 trials: {np.average(accs)*100:.2f}%±{np.std(accs)*100:.2f}')

tqdm_object = tqdm(train_loader, total=len(train_loader))

triclip = CLIPModel().to(CFG.device)
triclip.load_state_dict(torch.load('./50000t_50e_best.pt'))
triclip.eval()

embedding_stances = pd.DataFrame(columns=[i for i in range(257)])

for batch in tqdm_object:
    users = [path.split('/')[7] for path in batch["video_path"]]

    batch = {k: v.to(CFG.device) for k, v in batch.items() if k != 'video_path'}
    batch_embeddings = triclip.embed(batch)

    for i, user in enumerate(users):
        for e_i in range(3):
            try:
                embedding_stances = pd.concat([pd.DataFrame([[float(x) for x in batch_embeddings[e_i][i].cpu()] + [stance_map[user]]],
                                                columns=embedding_stances.columns),
                                                embedding_stances], 
                                                ignore_index=True)
            except:
                pass



# %%
X = embedding_stances.iloc[:, 0:256].values.tolist()
y = embedding_stances[256]

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
# X = torch.tensor(X, dtype=torch.float32)
# print(len(X))
# print(len(y))

# y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

from sklearn.ensemble import RandomForestClassifier

# Initialize the classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_clf.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

from sklearn.naive_bayes import GaussianNB

# Initialize the classifier
nb_clf = GaussianNB()

# Train the classifier
nb_clf.fit(X_train, y_train)

# Make predictions
y_pred_nb = nb_clf.predict(X_test)

# Evaluate the model
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

from sklearn.svm import SVC

# Initialize the classifier
svm_clf = SVC(kernel='linear', random_state=42)

# Train the classifier
svm_clf.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_clf.predict(X_test)

# Evaluate the model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Random Forest
y_scores_rf = rf_clf.predict_proba(X_test)[:, 1]  # Probabilities of the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_scores_rf)
model_stats['Random Forest'] = (fpr, tpr)

# Naive Bayes
y_scores_nb = nb_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores_nb)
model_stats['Naive Bayes'] = (fpr, tpr)

# SVM (use decision function for SVM as it doesn't output probabilities by default)
y_scores_svm = svm_clf.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores_svm)
model_stats['SVM'] = (fpr, tpr)

plt.figure(dpi=300)
for model_name, shiz in model_stats.items():
    roc_auc = auc(shiz[0], shiz[1])
    plt.plot(shiz[0], shiz[1], label=f'{model_name} AUC={roc_auc:.2f}') # ROC curve = TPR vs FPR

plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.legend()
plt.title("Receiver Operating Characteristics")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
imgcat(plt.gcf())
