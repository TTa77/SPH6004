import pickle
import random
import time
import pandas as pd
import numpy as np
import pygraphviz as pgv
import warnings
import os
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from model_validator import *
warnings.filterwarnings("ignore")

task_start_time = time.localtime()
task_start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", task_start_time)
task_start_time_nospace = time.strftime("%Y%m%d%H%M%S", task_start_time)

data = pd.read_csv("../data/processed/data_aft_feat_eng.csv", header=0)

output_fig_dir = "../fig/RandomForest/{}".format(task_start_time_nospace)
if not os.path.exists(output_fig_dir):
    os.makedirs(output_fig_dir)

logfile = open("../log/RandomForest.log", "a+")
logfile.write("******Task start at {}******\n".format(task_start_time_str))

features = data.drop("aki", axis=1)
outcomes = data["aki"]

# initialize model
random_state = random.randint(0, 1000)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=random_state, n_jobs=-1)
logfile.write("Current model parameters: \n{}\n\n".format(rf_model.get_params()))

# split dataset
split_random_state = random.randint(0,1000)
train_feature, test_feature, train_outcome, test_outcome = train_test_split(features, outcomes, test_size=0.2,
                                                                            random_state=split_random_state)
logfile.write("Dataset split parameters: {{\"test_size\"=0.2, \"shuffle\"=True, "
              "\"random_state\"={random_state}}}\n".format(random_state=split_random_state))

# train random forest model
rf_model.fit(train_feature, train_outcome)
with open("../model/random_forest_{}.bin".format(task_start_time_nospace), "wb") as f:
    pickle.dump(rf_model, f)

estimator_counter = 1
if not os.path.exists(os.path.join(output_fig_dir, "estimators")):
    os.makedirs(os.path.join(output_fig_dir, "estimators"))
for tree_model in rf_model.estimators_:
    tree.export_graphviz(tree_model, out_file=os.path.join(
        output_fig_dir, "estimators", "estimator_{}.dot".format(estimator_counter)),
                         feature_names=train_feature.columns.values, class_names=["0", "1"],
                         filled=True, rounded=True)
    # tree_model_graphviz = open(os.path.join(output_fig_dir, "graphviz.dot"), "r").read()
    graph = pgv.AGraph(os.path.join(output_fig_dir, "estimators", "estimator_{}.dot".format(estimator_counter)), ranksep=5)
    graph.graph_attr["size"] = "500,2500"
    graph.draw(os.path.join(output_fig_dir, "estimators", "estimator_{}_pic.png".format(estimator_counter)),
               format="png", prog="dot")
    estimator_counter += 1

# validate model
predicted_outcome_train = rf_model.predict(train_feature)
predicted_score_train = rf_model.predict_proba(train_feature)
predicted_outcome = rf_model.predict(test_feature)
predicted_score = rf_model.predict_proba(test_feature)

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
fig.savefig(os.path.join(output_fig_dir,
                         "RF_{time}.png".
                         format(time=task_start_time_nospace)))

logfile.write("******Task end at {}******\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
logfile.close()