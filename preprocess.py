from typing import List, Set, Dict, Tuple, Union, DefaultDict
import pandas as pd
import numpy as np
import re
import string
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings("ignore")


class Preprocess:
    def __init__(self, df_path: str, ):
        """
        :param df_path: path of titanic dataframe kind data with columns:
        ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
        'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        """
        self.df = pd.read_csv(df_path)
        self.features = list(self.df.columns)
        self.df = self.feature_engineering(self.df)
        print(self.df)
        print(self.features)

    def drop_small_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        nulls_percent = df.isna().sum() / len(df)
        # dropping row for features with less than 0.5% of missing values
        small_missing_values_features = [feature for feature in self.features if 0 < nulls_percent[feature] < 0.005]
        for feature in small_missing_values_features:
            df = df[df[feature].notnull()]
        return df

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

    @ staticmethod
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

    @staticmethod
    def feature_index_for_missing_data(df: pd.DataFrame, feature: str) -> None:
        # a binary column indicating which rows have missing data for that feature
        missing_feature_name = f"{feature}NaN"
        df[missing_feature_name] = np.where(df[feature].isnull(), 1, 0)

    def feature_engineering(self, df: pd.DataFrame):
        # dropping rows where Embarked is missing (in train just 2 rows)
        df = self.drop_small_missing_data(df)  # after that we are with column that have more than a few missing values

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
