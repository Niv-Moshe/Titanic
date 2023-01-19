from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class PassthroughTransformer(BaseEstimator):
    def fit(self, X, y=None):
        self.cols = X.columns
        return self

    def transform(self, X, y=None):
        self.cols = X.columns
        return X.values

    def get_feature_names_out(self):
        return np.array(self.cols)
