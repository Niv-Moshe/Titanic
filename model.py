from preprocess import Preprocess
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from feature_engine.selection import DropConstantFeatures
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression


class ModelTitanic:
    def __init__(self, df_train_path: str = None, ):
        """
        Will make a preprocessor sklearn ColumnTransformer for using in pipeline
        Args:
            df_train_path: path of titanic dataframe kind data with columns:
            ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        """
        self.df_train_path = df_train_path
        # self.df_train = pd.read_csv(self.df_train_path)
        preprocess = Preprocess(df_path=self.df_train_path)
        self.df_train = preprocess.get_df()
        self.features = list(self.df_train.columns)
        self.pipeline = preprocess.get_pipeline()
        self.pipeline.steps.append(['poly', PolynomialFeatures(include_bias=False, interaction_only=True)])
        # dropping interaction features that are columns of zeros
        self.pipeline.steps.append(['drop_constant_features', DropConstantFeatures()])
        self.pipeline.steps.append(['feature_select',
                                    RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='roc_auc')])
        self.pipeline.set_output(transform="pandas")
        X = self.pipeline.fit_transform(self.df_train, self.df_train['Survived'])
        print()
        print(X)
        print(X.shape)
