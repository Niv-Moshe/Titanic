import pandas as pd
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from feature_engine.selection import DropFeatures

import src.dataprep.preprocess_utils as preprocess_utils
from src.dataprep.FixNamesTransformer import FixNamesTransformer
warnings.filterwarnings("ignore")


class Preprocess:
    def __init__(self, df_train: pd.DataFrame = None, label_encode: bool = True, one_hot_encode: bool = True):
        """
        Will make a feature engineering and preprocessor sklearn pipeline.
        Args:
            df_train: titanic dataframe kind data with columns:
            ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
            label_encode: flag for performing label encoding (not necessary for lightgbm and catboost).
            one_hot_encode: flag for performing one hot encoding (not necessary for lightgbm and catboost).
        """
        assert df_train is not None, 'train dataframe shouldn\'t be None for preprocess class'
        self._df_train = df_train
        self._features = list(self._df_train.columns)
        self._label_encode = label_encode
        self._one_hot_encode = one_hot_encode
        # creating an empty pipeline
        self._preprocess_pipeline = Pipeline(steps=[], verbose=True)
        # updating the pipeline
        self.make_preprocess_pipeline()

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train

    @property
    def label_encode(self) -> bool:
        return self._label_encode

    @property
    def one_hot_encode(self) -> bool:
        return self._one_hot_encode

    @property
    def preprocess_pipeline(self) -> Pipeline:
        return self._preprocess_pipeline

    def feature_engineering(self) -> ColumnTransformer:
        """
        Function for making new features out of existing ones, and dealing with missing data partially.

        Returns:
            ColumnTransformer for creating new features.
        """
        nulls_count = self._df_train.isna().sum()
        features_with_missingness = [feature for feature in self._features if nulls_count[feature] >= 1]
        # after that we are with columns that have more than a few missing values
        self._df_train = preprocess_utils.drop_small_missing_data(self._df_train, features_with_missingness)

        transformers = [
            preprocess_utils.honorifics_function_transformer(['Name']),
            preprocess_utils.solo_function_transformer(['Parch', 'SibSp']),
            preprocess_utils.cabin_char_function_transformer(['Cabin']),
            preprocess_utils.ticket_group_and_group_size_function_transformer(['Ticket']),
            preprocess_utils.feature_index_for_missing_data_function_transformer(features_with_missingness)
        ]
        feature_engineering_transformer = ColumnTransformer(remainder='passthrough', transformers=transformers,
                                                            verbose_feature_names_out=False)
        return feature_engineering_transformer

    def make_preprocess_pipeline(self):
        """
        Appending new pipeline steps for feature engineering and preprocessing data.

        Returns: updates the self.pipeline object.
        """
        # dropping PassengerId which is an index column
        self._preprocess_pipeline.steps.append(['drop_PassengerId', DropFeatures(['PassengerId'])])

        # performing feature engineering - creating new features and droppping unncessary ones
        feature_engineering_transformer = self.feature_engineering()
        self._preprocess_pipeline.steps.append(['feature_engineering', feature_engineering_transformer])
        self._preprocess_pipeline.steps.append([
            'drop_unnecessary_columns',
            # two cabin columns are left because used twice in feature engineering
            DropFeatures(['Name', 'Ticket', 'Cabin'])
        ])

        # preprocessing the data: label encoding, one hot encoding and imputing
        transformers = []
        if self._label_encode:
            transformers.append(preprocess_utils.label_encoding_transformer(['Sex']))
        if self._one_hot_encode:
            transformers.append(preprocess_utils.one_hot_encoding_transformer(["Embarked", "Honorifics", "CabinChar"]))

        transformers.append(preprocess_utils.numeric_imputing_transformer(["Age"]))

        self._preprocess_pipeline.steps.append(['preprocess',
                                                ColumnTransformer(transformers=transformers, remainder='passthrough',
                                                                  verbose_feature_names_out=False)])
        self._preprocess_pipeline.set_output(transform="pandas")
