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


# Read data
train_df, valid_df, test_df = make_train_valid_dfs()
print(len(train_df))

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
train_loader = build_loaders(train_df, tokenizer, mode="train")
# valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

tqdm_object = tqdm(train_loader, total=len(train_loader))

models = {
    'triCLIP-100': './100t_50e_best.pt',
    'triCLIP-1k': './1000t_50e_best.pt',
    'triCLIP-10k': './10000t_50e_best.pt',
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
    # print(u_counts)

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

    # Convert to 2D PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    # print(len(X))
    # print(len(y))

    y = torch.tensor(y_resampled, dtype=torch.float32)#.reshape(-1, 1)
    # print(y)


        
    model = DeepMulticlass().to(CFG.device)

    # # train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

    # loss metric and optimizer
    # model = Multiclass()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # prepare model and training parameters
    n_epochs = 300
    batch_size = 1024
    # batches_per_epoch = len(X_train) // batch_size
    batch_starts = torch.arange(0, len(X_train), batch_size)
    # print(len(X_train))
    # print(batch_starts)
    
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
        for i in batch_starts:
            
            X_batch = X_train[i:i+batch_size].to(CFG.device)
            y_batch = y_train[i:i+batch_size]
            
            y_pred = model(X_batch).cpu()
            
            loss = loss_fn(y_pred, y_batch.type(torch.long))
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            
            acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            # print(f'Loss: {loss}, Acc: {acc}')
            
        # set model in evaluation mode and run through the test set
        model.eval()
        y_pred = model(X_test.to(CFG.device)).cpu()
        
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
        # print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

    # set model in evaluation mode and run through the test set
    model.eval()
    y_pred = model(X_test.to(CFG.device)).cpu()

    ce = loss_fn(y_pred, y_test.type(torch.long))
    acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
    ce = float(ce)
    acc = float(acc)
    print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")
    
    # Restore best model
    model.load_state_dict(best_weights)
    
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    import numpy as np
    import matplotlib.pyplot as plt

    # Assuming y_test is your true labels, and y_score is the score matrix from a classifier
    # Since you've used SMOTE, ensure y_test is appropriately processed for evaluation

    # Binarize the output
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_resampled))
    n_classes = y_test_binarized.shape[1]

    # Predict probabilities or decision function
    # Ensure to convert X_test to the appropriate format if it's not already
    model.eval()  # Make sure the model is in evaluation mode
    with torch.no_grad():
        y_score = model(torch.tensor(X_test, dtype=torch.float32).to(CFG.device)).cpu().numpy()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_fpr = np.unique(np.concatenate([np.linspace(0, 1, 100) for _ in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    model_stats[model_name] = (all_fpr, mean_tpr, auc(all_fpr, mean_tpr))
# %%
X = embedding_stances.iloc[:, 0:256].values.tolist()
y = embedding_stances[256]

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

n_neighbors = min(5, max(1, u_counts.most_common()[-1][1] - 1))  # Adjust based on your minimum class size

oversample = SMOTE(k_neighbors=n_neighbors)
X, y_resampled = oversample.fit_resample(X, y_encoded)
# Convert to 2D PyTorch tensors
# X = torch.tensor(X, dtype=torch.float32)
# print(len(X))
# print(len(y))

# y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

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

from sklearn.svm import SVC

# Initialize the classifier
svm_clf = SVC(kernel='linear', random_state=42)
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

    # # Plot the loss and accuracy
    # plt.plot(train_loss_hist, label="train")
    # plt.plot(test_loss_hist, label="test")
    # plt.xlabel("epochs")
    # plt.ylabel("cross entropy")
    # plt.legend()
    # plt.show()
    # imgcat(plt.gcf())
    # plt.clf()
    
    # plt.plot(train_acc_hist, label="train")
    # plt.plot(test_acc_hist, label="test")
    # plt.xlabel("epochs")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.show()
    # imgcat(plt.gcf())