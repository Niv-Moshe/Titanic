from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class PassthroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # y to support API
        self.cols = X.columns
        return self

    def transform(self, X, y=None):
        self.cols = X.columns
        return X.values

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names(self, input_features=None):
        # to support get_feature_names (get_feature_names is deprecated in 1.0 and
        # will be removed in 1.2. Please use get_feature_names_out instead)
        return self.get_feature_names_out(input_features)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.cols)
        return input_features
