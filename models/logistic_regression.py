import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class LR:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=400, multi_class='ovr', n_jobs=1)

    def train_model(self, train_x, train_y, save_name=None):
        self.scaler = StandardScaler().fit(train_x)
        # temperature = 1
        # train_x /= temperature
        self.model.fit(self.scaler.transform(train_x), train_y)
        joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        self.scaler = StandardScaler().fit(test_x)
        return self.model.predict_proba(self.scaler.transform(test_x))

    def test_model_acc(self, test_x, test_y):
        # self.load_model(model)
        pred_y = self.model.predict(self.scaler.transform(test_x))

        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler.transform(test_x))
        # return roc_auc_score(test_y, pred_y[:, 1])  # binary class classification AUC
        return roc_auc_score(test_y, pred_y[:, 1], multi_class="ovr", average=None)  # multi-class AUC