from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import copy
import random
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import stance_config as CFG

from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer

from imgcat import imgcat
from stances import stance_map
 
from triclip import CLIPModel
from utils import make_train_valid_dfs, build_loaders

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Binary target: 1 for Iris-setosa, 0 otherwise

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

triclip = CLIPModel().to(CFG.device)
triclip.load_state_dict(torch.load('./100t_50e_best.pt'))
triclip.eval()

# %%
# Read data
train_df, valid_df, test_df = make_train_valid_dfs()
users = [path.split('/')[7] for path in train_df["video_path"]]
print(len(train_df))
print(set(users))

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
train_loader = build_loaders(train_df, tokenizer, mode="train")
# valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

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
