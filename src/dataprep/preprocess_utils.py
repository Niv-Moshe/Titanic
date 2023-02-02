import re
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer  # IterativeImputer doesn't work without this import
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import FunctionTransformer

from src.consts import PreprocessUtilsConsts


def drop_small_missing_data(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Go over the features with missing values that have very little rows with missing values and will drop them.
    E.g. dropping rows for features with less than 0.5% of missing values.

    Args:
        df: dataframe to remove rows with small missing values.
        features: list of features of the dataframe.

    Returns:
        New dataframe with a rows of small missing data features removed.
    """
    nulls_percent = df.isna().sum() / len(df)
    small_missing_values_features = [feature for feature in features if 0 < nulls_percent[feature] <
                                     PreprocessUtilsConsts.MISSINGNESS_PCT]
    for feature in small_missing_values_features:
        df = df[df[feature].notnull()]
    return df


def feature_index_for_missing_data(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Create a new binary column indicating which rows have missing data for each feature.

    Args:
        df: dataframe to add the columns indicating where missing data of features is.
        features: list of features of the dataframe.

    Returns:
        New dataframe with a missing data binary columns.
    """
    nulls_count = df.isna().sum()
    for feature in features:
        if nulls_count[feature] >= 1:
            missing_feature_name = f"{feature}NaN"
            df[missing_feature_name] = np.where(df[feature].isnull(), 1, 0)
    return df


def find_honorifics(name: str) -> str:
    """
    Extract the honorifics title for a given name.

    Args:
        name: name string to extract the honorifics title from.

    Returns:
        Honorifics title string.
    """
    honorifics_list = PreprocessUtilsConsts.HONORIFICS_LIST
    honorifics_for_regex = "|".join(honorifics_list)
    honorific_names = rf'(?:{honorifics_for_regex})\.?'
    honorific = re.findall(honorific_names, name)
    if not honorific:  # empty list - no honorific found
        return 'non_honorific'
    honorific = honorific[0]  # findall returns a list - in our case list of single string
    honorific = honorific.split()[0]  # the string contains the honorific in first place if exists
    honorific = honorific.replace('.', '')
    return honorific


def honorifics_feature_from_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column with the honorifics title for each passenger.

    Args:
        df: dataframe to add the honorifics column in.

    Returns:
        New dataframe with a honorifics column.
    """
    df['Honorifics'] = df['Name'].apply(lambda name: find_honorifics(name))
    return df


def solo_feature_from_parch_sibsp(df) -> pd.DataFrame:
    """
    Create a new column which indicates if a passenger is traveling "solo".
    We say a passenger is travelling solo if both SibSp and Parch are 0.
    Could be traveling with friends but we only consider relatives here,
    friends could be captured in the group ticket feature.

    Args:
        df: dataframe to add the Honorifics column in.

    Returns:
        New dataframe with a Honorifics column.
    """
    df['Solo'] = (df['Parch'] + df['SibSp']).apply(lambda x: 1 if x == 0 else 0)
    return df


def cabin_char_feature_from_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column which holds the first character from the cabin of a passenger
    (cabin is with about 70% of missing values in train).

    Args:
        df: dataframe to add the CabinChar column in.

    Returns:
        New dataframe with a CabinChar column.
    """
    cabins = df['Cabin'].fillna(PreprocessUtilsConsts.CABIN_NONE_CONST)
    cabin_char = [cab[0] for cab in cabins]
    df['CabinChar'] = cabin_char
    return df


def ticket_group_and_group_size_features_from_ticket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns about information on tickets: Whether it is a group ticket (meaning some
    passengers have the same ticket) and the group size if it is a group ticket.

    Args:
        df: dataframe to add TicketGroup and GroupSize columns in.

    Returns:
        New dataframe with TicketGroup and GroupSize columns.
    """
    ticket_counts_dict = df['Ticket'].value_counts().to_dict()
    df['TicketGroup'] = df['Ticket'].apply(lambda x: 1 if ticket_counts_dict[x] > 1 else 0)
    df['GroupSize'] = df['Ticket'].map(ticket_counts_dict)
    return df


def label_encoding_transformer(label_encoder_features: List[str]) -> Tuple[str, OrdinalEncoder, List[str]]:
    """
    Return a tuple of a transformer (OrdinalEncoder) with its name to be in the ColumnTransformer and
    the features it will work on. Following the tuple format needed of transformers in ColumnTransformer.

    Args:
        label_encoder_features: list of features to perform label encoding on.

    Returns:
        Tuple following the format of transformers in ColumnTransformer.
    """
    label_encoder_transformer = OrdinalEncoder()
    return "label_encoder", label_encoder_transformer, label_encoder_features


def one_hot_encoding_transformer(one_hot_features: List[str]) -> Tuple[str, OneHotEncoder, List[str]]:
    """
    Return a tuple of a transformer (OneHotEncoder) with its name to be in the ColumnTransformer and
    the features it will work on. Following the tuple format needed of transformers in ColumnTransformer.

    Args:
        one_hot_features: list of features to perform one hot encoding on.

    Returns:
        Tuple following the format of transformers in ColumnTransformer.
    """
    one_hot_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return "onehot", one_hot_transformer, one_hot_features


def numeric_imputing_transformer(numeric_features: List[str]) -> Tuple[str, IterativeImputer, List[str]]:
    """
    Return a tuple of a transformer (IterativeImputer) with its name to be in the ColumnTransformer and
    the features it will work on. Following the tuple format needed of transformers in ColumnTransformer.

    Args:
        numeric_features: list of features to perform imputing on.

    Returns:
        Tuple following the format of transformers in ColumnTransformer.
    """
    numeric_transformer = IterativeImputer(estimator=LinearRegression(), missing_values=np.nan, max_iter=10,
                                           imputation_order='roman', random_state=0)
    return "numeric", numeric_transformer, numeric_features


def honorifics_function_transformer(name_feature: List[str]) -> Tuple[str, FunctionTransformer, List[str]]:
    """
    Return a tuple of a transformer (FunctionTransformer of honorifics_feature_from_name) with its name to be in the
    ColumnTransformer and the features it will work on (['Name']).
    Following the tuple format needed of transformers in ColumnTransformer.

    Args:
        name_feature: list of features needed for creating Honorifics. Should be just Name feature in a list - ['Name']

    Returns:
        Tuple following the format of transformers in ColumnTransformer.
    """
    return 'honorifics_feature_from_name', \
           FunctionTransformer(honorifics_feature_from_name, validate=False), \
           name_feature


def solo_function_transformer(parch_sibsp_features: List[str]) -> Tuple[str, FunctionTransformer, List[str]]:
    """
    Return a tuple of a transformer (FunctionTransformer of solo_feature_from_parch_sibsp) with its name to be in the
    ColumnTransformer and the features it will work on.
    Following the tuple format needed of transformers in ColumnTransformer.

    Args:
        parch_sibsp_features: list of features needed for creating Solo. Should be - ['Parch', 'SibSp'].

    Returns:
        Tuple following the format of transformers in ColumnTransformer.
    """
    return 'solo_feature_from_parch_sibsp', \
           FunctionTransformer(solo_feature_from_parch_sibsp, validate=False), \
           parch_sibsp_features


def cabin_char_function_transformer(cabin_feature: List[str]) -> Tuple[str, FunctionTransformer, List[str]]:
    """
    Return a tuple of a transformer (FunctionTransformer of cabin_char_feature_from_cabin) with its name to be in the
    ColumnTransformer and the features it will work on.
    Following the tuple format needed of transformers in ColumnTransformer.

    Args:
        cabin_feature: list of features needed for creating CabinChar. Should be - ['Cabin].

    Returns:
        Tuple following the format of transformers in ColumnTransformer.
    """
    return 'CabinChar_feature_from_cabin', \
           FunctionTransformer(cabin_char_feature_from_cabin, validate=False), \
           cabin_feature


def ticket_group_and_group_size_function_transformer(ticket_feature: List[str]) -> \
        Tuple[str, FunctionTransformer, List[str]]:
    """
    Return a tuple of a transformer (FunctionTransformer of ticket_group_and_group_size_features_from_ticket) with its
    name to be in the ColumnTransformer and the features it will work on.
    Following the tuple format needed of transformers in ColumnTransformer.

    Args:
        ticket_feature: list of features needed for creating TicketGroup and GroupSize. Should be - ['Ticket].

    Returns:
        Tuple following the format of transformers in ColumnTransformer.
    """
    return 'TicketGroup_GroupSize_features_from_ticket', \
           FunctionTransformer(ticket_group_and_group_size_features_from_ticket, validate=False), \
           ticket_feature


def feature_index_for_missing_data_function_transformer(features_with_missingness: List[str]) -> \
        Tuple[str, FunctionTransformer, List[str]]:
    """
    Return a tuple of a transformer (FunctionTransformer of feature_index_for_missing_data) with its
    name to be in the ColumnTransformer and the features it will work on.
    Following the tuple format needed of transformers in ColumnTransformer.

    Args:
        features_with_missingness: list of features needed to create a binary column indicating missing data.

    Returns:
        Tuple following the format of transformers in ColumnTransformer.
    """
    return 'feature_index_for_missing_data', \
           FunctionTransformer(feature_index_for_missing_data,
                               kw_args={'features': features_with_missingness}, validate=False), \
           features_with_missingness
