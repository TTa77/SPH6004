from typing import Iterable, Optional
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes


class ModelValidator(object):

    def __init__(self, y_actual: Iterable, y_predict: Iterable, y_score: Iterable):
        self.y_actual = y_actual
        self.y_predict = y_predict
        self.y_score = y_score

        self._tp = None
        self._fp = None
        self._tn = None
        self._fn = None

        self._pcs = None
        self._rec = None

        self._f1 = None
        self._roc_auc = None

        self.update_model_effect()

    def update_model_effect(self):
        self._tp = self.calc_true_positive()
        self._fp = self.calc_false_positive()
        self._tn = self.calc_true_negative()
        self._fn = self.calc_false_negative()

        self._pcs = self.calc_precision()
        self._rec = self.calc_recall()
        self._f1 = self.calc_f1_score()
        self._roc_auc = self.calc_roc_auc()

    def calc_f1_score(self):
        f1 = f1_score(self.y_actual, self.y_predict)
        return f1

    def f1_score(self):
        return self._f1

    def draw_pr(self, fig: Optional[Axes], color="b", draw_random=True):
        precision, recall, thresholds = precision_recall_curve(self.y_actual, self.y_score)
        if fig is None:
            fig = plt.figure()
            if draw_random:
                plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)
            plt.plot(recall, precision, marker='o', markersize=2,
                     lw=2, label='F1 Score= {:.2f}'.format(self._f1), color=color, alpha=.8)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='best')
            plt.grid(visible=True)
        else:
            if draw_random:
                fig.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)
            fig.plot(recall, precision, marker='o', lw=2, markersize=2,
                     label='F1 Score= {:.2f}'.format(self._f1), color=color, alpha=.8)
            fig.set_xlabel('Recall')
            fig.set_ylabel('Precision')
            fig.set_ylim([-0.05, 1.05])
            fig.set_title('Precision-Recall Curve')
            fig.legend(loc='best')
            fig.grid(visible=True)
        return fig

    def calc_precision(self):
        try:
            pcs = self._tp / (self._tp + self._fp)
        except ZeroDivisionError:
            pcs = 1
        return pcs

    def calc_recall(self):
        try:
            rec = self._tp / (self._tp + self._fn)
        except ZeroDivisionError:
            rec = 1
        return rec

    def calc_roc_auc(self):
        roc_auc = roc_auc_score(self.y_actual, self.y_score)
        return roc_auc

    def draw_roc_auc(self, fig: Optional[Axes], color="b", draw_random=True):
        fpr, tpr, thresholds = roc_curve(y_true=self.y_actual, y_score=self.y_score)

        if fig is None:
            fig = plt.figure()
            plt.plot(fpr, tpr, color=color, lw=2, label='ROC curve (area = {:.2f})'.format(self._roc_auc))
            if draw_random:
                plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='best')
            plt.grid(visible=True)
        else:
            fig.plot(fpr, tpr, color=color, lw=2, label='ROC curve (area = {:.2f})'.format(self._roc_auc))
            if draw_random:
                fig.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)
            fig.set_xlabel('False Positive Rate')
            fig.set_ylabel('True Positive Rate')
            fig.set_title('Receiver Operating Characteristic (ROC)')
            fig.legend(loc='best')
            fig.grid(visible=True)
        return fig

    def calc_specificity(self):
        fpr = self._fp / (self._fp + self._tn)
        return 1 - fpr

    def calc_sensitivity(self):
        return self.calc_recall()

    def calc_accuracy(self):
        correct_counter = 0
        for ya, yp in zip(self.y_actual, self.y_predict):
            if ya == yp:
                correct_counter += 1

        acc = correct_counter / len(self.y_actual)

        return acc

    def calc_true_positive(self):
        tp = 0
        for ya, yp in zip(self.y_actual, self.y_predict):
            if ya == 1 and yp == 1:
                tp += 1
        return tp

    def calc_true_negative(self):
        tn = 0
        for ya, yp in zip(self.y_actual, self.y_predict):
            if ya == 0 and yp == 0:
                tn += 1
        return tn

    def calc_false_positive(self):
        fp = 0
        for ya, yp in zip(self.y_actual, self.y_predict):
            if ya == 0 and yp == 1:
                fp += 1
        return fp

    def calc_false_negative(self):
        fn = 0
        for ya, yp in zip(self.y_actual, self.y_predict):
            if ya == 1 and yp == 0:
                fn += 1
        return fn

