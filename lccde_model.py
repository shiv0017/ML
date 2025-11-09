# lccde_model.py
import numpy as np
from statistics import mode

class LCCDE_Ensemble:
    def __init__(self, m1, m2, m3, model_leaders):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.model_leaders = model_leaders

    def predict(self, X):
        yp = []
        for _, xi in X.iterrows():
            xi2 = np.array(list(xi.values))
            y_pred1 = int(self.m1.predict(xi2.reshape(1, -1))[0])
            y_pred2 = int(self.m2.predict(xi2.reshape(1, -1))[0])
            y_pred3 = int(self.m3.predict(xi2.reshape(1, -1))[0])

            p1 = self.m1.predict_proba(xi2.reshape(1, -1))
            p2 = self.m2.predict_proba(xi2.reshape(1, -1))
            p3 = self.m3.predict_proba(xi2.reshape(1, -1))

            y_pred_p1 = np.max(p1)
            y_pred_p2 = np.max(p2)
            y_pred_p3 = np.max(p3)

            if y_pred1 == y_pred2 == y_pred3:
                y_pred = y_pred1
            elif y_pred1 != y_pred2 != y_pred3:
                l, pred_l, pro_l = [], [], []
                if self.model_leaders[y_pred1] == self.m1:
                    l.append(self.m1); pred_l.append(y_pred1); pro_l.append(y_pred_p1)
                if self.model_leaders[y_pred2] == self.m2:
                    l.append(self.m2); pred_l.append(y_pred2); pro_l.append(y_pred_p2)
                if self.model_leaders[y_pred3] == self.m3:
                    l.append(self.m3); pred_l.append(y_pred3); pro_l.append(y_pred_p3)

                if len(l) == 0:
                    pro_l = [y_pred_p1, y_pred_p2, y_pred_p3]
                if len(l) == 1:
                    y_pred = pred_l[0]
                else:
                    max_p = max(pro_l)
                    if max_p == y_pred_p1: y_pred = y_pred1
                    elif max_p == y_pred_p2: y_pred = y_pred2
                    else: y_pred = y_pred3
            else:
                n = mode([y_pred1, y_pred2, y_pred3])
                y_pred = int(self.model_leaders[n].predict(xi2.reshape(1, -1))[0])
            yp.append(y_pred)
        return yp
