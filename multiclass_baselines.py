# %%
import copy
import random

from imblearn.over_sampling import SMOTE
 
from sklearn.metrics import accuracy_score
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

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearnex import patch_sklearn 

patch_sklearn()

def train_predict_plot(classifier, classifier_name, X_train, X_test, y_train, y_test):
    # Fit the classifier
    classifier.fit(X_train, y_train)
    
    # Predict probabilities or decision function
    if hasattr(classifier, "predict_proba"):
        y_scores = classifier.predict_proba(X_test)
    else:  # Use decision function for models like SVM
        y_scores = classifier.decision_function(X_test)
        # Convert scores to probabilities using softmax
        y_scores = np.exp(y_scores) / np.sum(np.exp(y_scores), axis=1, keepdims=True)
    
    # Binarize the output labels for multiclass ROC
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
    n_classes = y_test_binarized.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Average it and compute AUC
    mean_tpr /= n_classes

    y_pred = classifier.predict(X_test)
    print(f"{classifier_name} Accuracy:", accuracy_score(y_test, y_pred))

    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)


# Read data
train_df, valid_df, test_df = make_train_valid_dfs(10000)
print(len(train_df))

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
train_loader = build_loaders(train_df, tokenizer, mode="train")
# valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

tqdm_object = tqdm(train_loader, total=len(train_loader))

models = {
    'triCLIP-50k': './50000t_50e_best.pt'
}

model_stats = {}

for model_name, model_path in models.items():
    triclip = CLIPModel().to(CFG.device)
    triclip.load_state_dict(torch.load(model_path))
    triclip.eval()

    embedding_stances = pd.DataFrame(columns=[i for i in range(257)])

    for batch in tqdm_object:
        users = [path.split('/')[7] for path in batch["video_path"]]

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
# print(u_counts)

u = [[username] for username in y]
y = u

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)

n_neighbors = min(5, max(1, u_counts.most_common()[-1][1] - 1))  # Adjust based on your minimum class size

oversample = SMOTE(k_neighbors=n_neighbors)
X, y_resampled = oversample.fit_resample(X, y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_resampled, train_size=0.8, shuffle=True)

from sklearn.ensemble import RandomForestClassifier

# Initialize the classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
a, b, c = train_predict_plot(rf_clf, "Random Forest", X_train, X_test, y_train, y_test)
model_stats['Random Forest'] = (a, b, c)

from sklearn.naive_bayes import GaussianNB

# Initialize the classifier
nb_clf = GaussianNB()
a, b, c = train_predict_plot(nb_clf, "Naive Bayes", X_train, X_test, y_train, y_test)
model_stats['Naive Bayes'] = (a, b, c)

from sklearn.svm import LinearSVC

# Initialize the classifier
svm_clf = LinearSVC(random_state=42, max_iter=1000)
a, b, c = train_predict_plot(svm_clf, "SVM", X_train, X_test, y_train, y_test)
model_stats['SVM'] = (a, b, c)

# Plot the macro-averaged ROC curve
plt.figure(dpi=300)
for model_name, stats in model_stats.items():
    plt.plot(stats[0], stats[1], label=f'{model_name} (AUC={stats[2]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Averaged ROC Curve')
plt.legend(loc="lower right")
plt.show()
imgcat(plt.gcf())