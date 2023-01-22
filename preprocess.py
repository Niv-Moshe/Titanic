from PassthroughTransformer import PassthroughTransformer
from ColumnTransformerWithNames import ColumnTransformerWithNames
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer  # IterativeImputer doesn't work without this import
from sklearn.impute import IterativeImputer

import warnings
warnings.filterwarnings("ignore")


class Preprocess:
    def __init__(self, df: pd.DataFrame = None, df_path: str = None):
        """
        Will make a preprocessor sklearn ColumnTransformer for using in pipeline
        Args:
            df_path: path of titanic dataframe kind data with columns:
            ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
            df: if given otherwise read from path
        """
        assert not all(v is None for v in [df_path, df]), "One argument should be not None"
        self.transformers = []
        if df_path is not None:
            self.df = pd.read_csv(df_path)
        else:
            self.df = df
        self.features = list(self.df.columns)
        self.df = self.feature_engineering(self.df)
        self.preprocessor = self.make_transformers()

    def get_df(self):
        return self.df

    def get_preprocessor(self):
        return self.preprocessor

    def drop_small_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        nulls_percent = df.isna().sum() / len(df)
        # dropping row for features with less than 0.5% of missing values
        small_missing_values_features = [feature for feature in self.features if 0 < nulls_percent[feature] < 0.005]
        for feature in small_missing_values_features:
            df = df[df[feature].notnull()]
        return df

    @staticmethod
    def feature_index_for_missing_data(df: pd.DataFrame, feature: str) -> None:
        # a binary column indicating which rows have missing data for that feature
        if feature == 'Survived':  # wouldn't happen (full data on independent variable)
            return
        missing_feature_name = f"{feature}NaN"
        df[missing_feature_name] = np.where(df[feature].isnull(), 1, 0)

    @staticmethod
    def find_honorifics(name: str) -> str:
        honorific_names = r'(?:Mrs|Mr|Ms|Miss|Master|Don|Rev|Mme|Major|Mlle|Col|Capt|Jonkheer|Countess|Dr)\.?'
        honorific = re.findall(honorific_names, name)
        if not honorific:  # empty list - no honorific found
            return 'non_honorific'
        honorific = honorific[0]  # findall returns a list - in our case list of single string
        honorific = honorific.split()[0]  # the string contains the honorific in first place if exists
        honorific = honorific.replace('.', '')
        return honorific

    def honorifics_feature_from_name(self, df: pd.DataFrame) -> None:
        df['Honorifics'] = df['Name'].apply(lambda name: self.find_honorifics(name))

    @staticmethod
    def solo_feature_from_parch_sibsp(df) -> None:
        # If both SibSp and Parch are 0 then we say the passenger is by himself on board, meaning he is traveling solo
        df['Solo'] = (df['Parch'] + df['SibSp']).apply(lambda x: 1 if x == 0 else 0)

    @staticmethod
    def CabinChar_feature_from_cabin(df: pd.DataFrame) -> None:
        # Extracting first character from the cabin (which is with about 70% of missing values in train)
        cabins = df['Cabin'].astype(str)
        cabin_char = [cab[0] for cab in cabins]  # when NaN then char would be 'n'
        df['CabinChar'] = cabin_char

    @staticmethod
    def TicketGroup_GroupSize_features_from_ticket(df: pd.DataFrame) -> None:
        ticket_counts_dict = df['Ticket'].value_counts().to_dict()
        df['TicketGroup'] = df['Ticket'].apply(lambda x: 1 if ticket_counts_dict[x] > 1 else 0)
        df['GroupSize'] = df['Ticket'].map(ticket_counts_dict)
        pass

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for making new features out of existing ones, and dealing with missing data partially
        Args:
            df: dataframe to process, will be with columns as specified in titanic

        Returns: a new dataframe with the new features and redundant features are dropped
        """
        # dropping rows where Embarked is missing (in train just 2 rows)
        df = self.drop_small_missing_data(df)  # after that we are with columns that have more than a few missing values

        # Features for when we have missing data
        nulls_count = df.isna().sum()
        for feature in self.features:
            if nulls_count[feature] >= 1:
                self.feature_index_for_missing_data(df, feature)

        # creating new features out of exising ones
        self.honorifics_feature_from_name(df)
        self.solo_feature_from_parch_sibsp(df)
        self.CabinChar_feature_from_cabin(df)
        self.TicketGroup_GroupSize_features_from_ticket(df)
        # dropping unnecessary column
        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        self.features = list(df.columns)
        return df

    def make_transformers(self, ) -> ColumnTransformerWithNames:
        non_transformed = ['Pclass', 'SibSp', 'Parch', 'Fare', 'TicketGroup', 'GroupSize', 'Solo']
        nan_features = [feature for feature in self.features if 'NaN' in feature]  # , 'AgeNaN', 'CabinNaN'
        non_transformed += nan_features

        label_encoder_features = ["Sex"]
        label_encoder_transformer = OrdinalEncoder()

        one_hot_features = ["Embarked", "Honorifics", "CabinChar"]
        one_hot_transformer = OneHotEncoder(handle_unknown="ignore")

        numeric_features = ["Age"]
        numeric_transformer = IterativeImputer(estimator=LinearRegression(), missing_values=np.nan, max_iter=10,
                                               imputation_order='roman', random_state=0)

        transformers = [
            ("label_encoder", label_encoder_transformer, label_encoder_features),
            ("onehot", one_hot_transformer, one_hot_features),
            ("numeric", numeric_transformer, numeric_features),
            # Survived column if exists will not make through the preprocessing (not in non_transformed)
            ('passthrough_transformer', PassthroughTransformer(), non_transformed),
        ]
        self.transformers += transformers
        return ColumnTransformerWithNames(transformers=self.transformers)
