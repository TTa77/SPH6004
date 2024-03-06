import random
import pandas as pd
import numpy as np
import warnings
import pickle
import time
import os
from sklearn import model_selection, linear_model
from sklearn.model_selection import train_test_split

from model_validator import *
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
# pd.set_option("display.max_rows", None)
np.set_printoptions(suppress=True)

task_start_time = time.localtime()
task_start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", task_start_time)
task_start_time_nospace = time.strftime("%Y%m%d%H%M%S", task_start_time)

data = pd.read_csv("../data/processed/data_aft_feat_eng.csv", header=0)
output_fig_dir = "../fig/LogisticRegression/{}".format(task_start_time_nospace)
if not os.path.exists(output_fig_dir):
    os.makedirs(output_fig_dir)

logfile = open("../log/LogisticRegression.log", "a+")
logfile.write("******Task start at {}******\n".format(task_start_time_str))

features = data.drop("aki", axis=1)
outcomes = data["aki"]

coefs = []
intercepts = []
logistic_model = linear_model.LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
logfile.write("Current model parameters: \n{}\n\n".format(logistic_model.get_params()))

kfold = 5
random_state = random.randint(0, 1000)
stratified_kfold_selector = model_selection.StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
logfile.write("KFold: {}  Random State: {}\n\n".format(kfold, random_state))
round_counter = 1

# kfold cross validation
for train, test in stratified_kfold_selector.split(X=features, y=outcomes):
    # oversampling, making non-aki : aki ratio to 0.8
    # smt = SMOTE(sampling_strategy=0.6)
    # train_feature, train_outcome = smt.fit_resample(features.iloc[train, :], outcomes[train].values)
    train_feature = features.iloc[train, :]
    train_outcome = outcomes[train].values
    test_feature = features.iloc[test, :]
    test_outcome = outcomes[test].values

    logistic_model.fit(train_feature, train_outcome)
    coef = logistic_model.coef_[0]
    intcp = logistic_model.intercept_
    logfile.write("Round {}:\n".format(round_counter))
    logfile.write("Coefficients:\n{}\n".format(coef))
    coefs.append(logistic_model.coef_[0])
    logfile.write("Intercept: {}\n\n".format(intcp))
    intercepts.append(intcp)

    predicted_outcome_train = logistic_model.predict(train_feature)
    predicted_score_train = logistic_model.predict_proba(train_feature)
    predicted_outcome = logistic_model.predict(test_feature)
    predicted_score = logistic_model.predict_proba(test_feature)

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
    fig.savefig(os.path.join(output_fig_dir,
                             "LR_{file_counter}_{time}.png".
                             format(file_counter=round_counter, time=task_start_time_nospace)))
    fig.show()
    round_counter += 1

# generate final model by mean values
mean_coef = np.mean(np.array(coefs), axis=0)
mean_intercept = np.mean(np.array(intercepts))
logfile.write("-----> Final Model\n")
logfile.write("Mean Coefficients:\n{}\n".format(mean_coef))
logfile.write("Mean Intercept: {}\n\n".format(mean_intercept))
logistic_model.coef_ = mean_coef.reshape(1,-1)
logistic_model.intercept_ = mean_intercept

with open("../model/logistic_regression_{}.bin".format(task_start_time_nospace), "wb") as f:
    pickle.dump(logistic_model, f)

# test final model
x_train, x_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=10)
predicted_score = logistic_model.predict_proba(x_test)
predicted_outcome = logistic_model.predict(x_test)
validator = ModelValidator(y_actual=y_test, y_predict=predicted_outcome, y_score=predicted_score[:, 1])
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
validator.draw_pr(axs[0], color="blue", draw_random=True)
validator.draw_roc_auc(axs[1], color="blue", draw_random=True)
fig.show()
fig.savefig(os.path.join(output_fig_dir,
                         "LR_final_{time}.png".
                         format(file_counter=round_counter, time=task_start_time_nospace)))

logfile.write("******Task end at {}******\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
logfile.close()
