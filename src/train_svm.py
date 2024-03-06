import pandas as pd
import numpy as np
import warnings
from sklearn import model_selection, svm
from model_validator import *

warnings.filterwarnings("ignore")
# pd.set_option("display.max_rows", None)

data = pd.read_csv("../data/processed/data_aft_feat_eng.csv", header=0)

features = data.drop("aki", axis=1)
outcomes = data["aki"]

stratified_kfold_selector = model_selection.StratifiedKFold(n_splits=5, shuffle=True)

for train, test in stratified_kfold_selector.split(X=features, y=outcomes):
    svm_classifier = svm.SVC(kernel="rbf", probability=True, max_iter=50000)

    train_feature = features.iloc[train, :]
    train_outcome = outcomes[train].values
    test_feature = features.iloc[test, :]
    test_outcome = outcomes[test].values

    svm_classifier.fit(train_feature, train_outcome)
    predicted_outcome_train = svm_classifier.predict(train_feature)
    predicted_score_train = svm_classifier.predict_proba(train_feature)
    predicted_outcome = svm_classifier.predict(test_feature)
    predicted_score = svm_classifier.predict_proba(test_feature)

    # validate model by training set and testing set respectively
    train_validator = ModelValidator(y_actual=train_outcome,
                                     y_predict=predicted_outcome_train, y_score=predicted_score_train[:, 1])
    test_validator = ModelValidator(y_actual=test_outcome,
                                    y_predict=predicted_outcome, y_score=predicted_score[:, 1])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    train_validator.draw_pr(axs[0], color="orange", draw_random=True)
    test_validator.draw_pr(axs[0], color="blue", draw_random=False)
    train_validator.draw_roc_auc(axs[1], color="orange", draw_random=True)
    test_validator.draw_roc_auc(axs[1], color="blue", draw_random=False)
    fig.show()
    break
