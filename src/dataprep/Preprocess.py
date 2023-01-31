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
    def __init__(self, df_train: pd.DataFrame = None):
        """
        Will make a feature engineering and preprocessor sklearn pipeline
        Args:
            df_train: titanic dataframe kind data with columns:
            ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        """
        assert df_train is not None, 'train dataframe shouldn\'t be None for preprocess class'
        self._df_train = df_train
        self.features = list(self._df_train.columns)
        # creating an empty pipeline
        self._preprocess_pipeline = Pipeline(steps=[], verbose=True)
        # updating the pipeline
        self.make_preprocess_pipeline()

    @property
    def df_train(self):
        return self._df_train

    @property
    def preprocess_pipeline(self):
        return self._preprocess_pipeline

    def feature_engineering(self) -> ColumnTransformer:
        """
        Function for making new features out of existing ones, and dealing with missing data partially
        """
        nulls_count = self._df_train.isna().sum()
        features_with_missingness = [feature for feature in self.features if nulls_count[feature] >= 1]
        # after that we are with columns that have more than a few missing values
        self._df_train = preprocess_utils.drop_small_missing_data(self._df_train, features_with_missingness)

        feature_engineering_transformer = ColumnTransformer(remainder='passthrough', transformers=[
            ('honorifics_feature_from_name', FunctionTransformer(preprocess_utils.honorifics_feature_from_name,
                                                                 validate=False),
             ['Name']),
            ('solo_feature_from_parch_sibsp', FunctionTransformer(preprocess_utils.solo_feature_from_parch_sibsp,
                                                                  validate=False),
             ['Parch', 'SibSp']),
            ('CabinChar_feature_from_cabin', FunctionTransformer(preprocess_utils.cabin_char_feature_from_cabin,
                                                                 validate=False),
             ['Cabin']),
            ('TicketGroup_GroupSize_features_from_ticket',
             FunctionTransformer(preprocess_utils.ticket_group_and_group_size_features_from_ticket, validate=False),
             ['Ticket']),
            ('feature_index_for_missing_data', FunctionTransformer(preprocess_utils.feature_index_for_missing_data,
                                                                   kw_args={'features': features_with_missingness},
                                                                   validate=False),
             features_with_missingness),
        ])
        return feature_engineering_transformer

    def make_preprocess_pipeline(self):
        """
        Appending new pipeline steps for feature engineering and preprocessing data

        Returns: updates the self.pipeline object
        """
        # dropping PassengerId which is an index column and target variable Survived
        self._preprocess_pipeline.steps.append(['drop_PassengerId_Survived', DropFeatures(['PassengerId', 'Survived'])])

        # performing feature engineering - creating new features and droppping unncessary ones
        feature_engineering_transformer = self.feature_engineering()
        self._preprocess_pipeline.steps.append(['feature_engineering', feature_engineering_transformer])
        self._preprocess_pipeline.steps.append(['fix_names_after_feature_engineering', FixNamesTransformer()])
        self._preprocess_pipeline.steps.append(['drop_unnecessary_columns',
                                                DropFeatures(['Name',
                                                             'Ticket',
                                                             'Cabin',  # two cabin columns are left because used twice
                                                             ])])

        # preprocessing the data: label encoding, one hot encoding and imputing
        transformers = [preprocess_utils.label_encoding_transformer(['Sex']),
                        preprocess_utils.one_hot_encoding_transformer(["Embarked", "Honorifics", "CabinChar"]),
                        preprocess_utils.numeric_imputing_transformer(["Age"])
                        ]
        self._preprocess_pipeline.steps.append(['preprocess',
                                                ColumnTransformer(transformers=transformers, remainder='passthrough')])
        self._preprocess_pipeline.steps.append(['fix_names_after_preprocess', FixNamesTransformer()])
        self._preprocess_pipeline.set_output(transform="pandas")
