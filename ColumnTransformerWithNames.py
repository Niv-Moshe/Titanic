import pandas as pd
from sklearn.compose import ColumnTransformer
from utils import get_feature_names_out


class ColumnTransformerWithNames(ColumnTransformer):
    def get_feature_names_out(self, input_features=None):
        return get_feature_names_out(self)

    def transform(self, X):
        indices = X.index.values.tolist()
        X_mat = super().transform(X)
        new_cols = self.get_feature_names_out()
        new_X = pd.DataFrame(X_mat, index=indices, columns=new_cols)
        return new_X

    def fit_transform(self, X, y=None):
        super().fit_transform(X, y)
        return self.transform(X)
