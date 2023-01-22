from utils import get_feature_names_out
from sklearn.pipeline import Pipeline
import pandas as pd


class PipelineWithNames(Pipeline):
    def get_feature_names_out(self, input_features=None):
        return get_feature_names_out(self)

    def transform(self, X):
        indices = X.index.values.tolist()
        new_cols = self.get_feature_names_out()
        X_mat = super().transform(X)
        new_X = pd.DataFrame(X_mat, index=indices, columns=new_cols)
        return new_X

    def fit_transform(self, X, y=None, **fit_params):
        super().fit_transform(X, y)
        return self.transform(X)
