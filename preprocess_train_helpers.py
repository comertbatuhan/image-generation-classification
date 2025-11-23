import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, log_loss
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle


def train_eval_logreg(
    X_train,
    X_test,
    y_train,
    y_test,
    n_epochs=100,
    learning_rate="optimal",
    random_state=42
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_test)
    classes = np.unique(y_train)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-2,
        learning_rate=learning_rate,
        random_state=random_state
    )

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        X_epoch, y_epoch = shuffle(
            X_train_scaled, y_train,
            random_state=random_state + epoch
        )

        if epoch == 0:
            clf.partial_fit(X_epoch, y_epoch, classes=classes)
        else:
            clf.partial_fit(X_epoch, y_epoch)

        y_train_proba = clf.predict_proba(X_train_scaled)
        y_val_proba   = clf.predict_proba(X_val_scaled)

        train_loss = log_loss(y_train, y_train_proba, labels=classes)
        val_loss   = log_loss(y_test,  y_val_proba,   labels=classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    epochs = np.arange(1, n_epochs + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='Train log-loss')
    plt.plot(epochs, val_losses, label='Validation log-loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log-loss')
    plt.title('SGD Logistic Regression Train / Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    y_pred = clf.predict(X_val_scaled)
    y_pred_proba = clf.predict_proba(X_val_scaled)

    acc = accuracy_score(y_test, y_pred)
    mf1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')

    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {mf1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")

    return clf, scaler



def build_X_num_only(df_in: pd.DataFrame, target_col='class'):
    feature_cols = [c for c in df_in.columns if c != target_col]
    X = df_in[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df_in[target_col].to_numpy()
    return X, y


def build_X_cat_num(
    df_in: pd.DataFrame,
    categorical_cols,
    numeric_cols,
    target_col='class',
):
    y = df_in[target_col].to_numpy()
    X_cat = pd.get_dummies(df_in[categorical_cols], drop_first=True, dtype=np.float32)
    X_num = df_in[numeric_cols].astype(np.float32)
    X = np.hstack([X_num.to_numpy(copy=False), X_cat.to_numpy(copy=False)])
    return X, y

