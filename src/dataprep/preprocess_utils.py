import re
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer  # IterativeImputer doesn't work without this import
from sklearn.impute import IterativeImputer

MISSINGNESS_PCT = 0.005


def drop_small_missing_data(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    nulls_percent = df.isna().sum() / len(df)
    # dropping row for features with less than 0.5% of missing values
    small_missing_values_features = [feature for feature in features if 0 < nulls_percent[feature] < MISSINGNESS_PCT]
    for feature in small_missing_values_features:
        df = df[df[feature].notnull()]
    return df


def feature_index_for_missing_data(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    nulls_count = df.isna().sum()
    for feature in features:
        if nulls_count[feature] >= 1:
            # a binary column indicating which rows have missing data for that feature
            if feature == 'Survived':  # wouldn't happen (full data on independent variable)
                continue
            missing_feature_name = f"{feature}NaN"
            df[missing_feature_name] = np.where(df[feature].isnull(), 1, 0)
    return df


def find_honorifics(name: str) -> str:
    honorific_names = r'(?:Mrs|Mr|Ms|Miss|Master|Don|Rev|Mme|Major|Mlle|Col|Capt|Jonkheer|Countess|Dr)\.?'
    honorific = re.findall(honorific_names, name)
    if not honorific:  # empty list - no honorific found
        return 'non_honorific'
    honorific = honorific[0]  # findall returns a list - in our case list of single string
    honorific = honorific.split()[0]  # the string contains the honorific in first place if exists
    honorific = honorific.replace('.', '')
    return honorific


def honorifics_feature_from_name(df: pd.DataFrame) -> pd.DataFrame:
    df['Honorifics'] = df['Name'].apply(lambda name: find_honorifics(name))
    return df


def solo_feature_from_parch_sibsp(df, y=None) -> pd.DataFrame:
    # If both SibSp and Parch are 0 then we say the passenger is by himself on board, meaning he is traveling solo
    df['Solo'] = (df['Parch'] + df['SibSp']).apply(lambda x: 1 if x == 0 else 0)
    return df


def cabin_char_feature_from_cabin(df: pd.DataFrame) -> pd.DataFrame:
    # Extracting first character from the cabin (which is with about 70% of missing values in train)
    cabins = df['Cabin'].astype(str)
    cabin_char = [cab[0] for cab in cabins]  # when NaN then char would be 'n'
    df['CabinChar'] = cabin_char
    return df


def ticket_group_and_group_size_features_from_ticket(df: pd.DataFrame) -> pd.DataFrame:
    ticket_counts_dict = df['Ticket'].value_counts().to_dict()
    df['TicketGroup'] = df['Ticket'].apply(lambda x: 1 if ticket_counts_dict[x] > 1 else 0)
    df['GroupSize'] = df['Ticket'].map(ticket_counts_dict)
    return df


def label_encoding_transformer(label_encoder_features):
    label_encoder_transformer = OrdinalEncoder()
    return "label_encoder", label_encoder_transformer, label_encoder_features


def one_hot_encoding_transformer(one_hot_features):
    one_hot_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return "onehot", one_hot_transformer, one_hot_features


def numeric_imputing_transformer(numeric_features):
    numeric_transformer = IterativeImputer(estimator=LinearRegression(), missing_values=np.nan, max_iter=10,
                                           imputation_order='roman', random_state=0)
    return "numeric", numeric_transformer, numeric_features
