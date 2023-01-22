# import warnings
import sklearn
import pandas as pd
import numpy as np


# Turn loopkup into function for better handling with pipeline later
def get_names(column_transformer_get, trans_get, columns_get):
    # >> Original get_feature_names() method
    # drop will be for Survived if provided
    if trans_get == 'drop' or (hasattr(columns_get, '__len__') and not len(columns_get)):
        return []
    if trans_get == 'passthrough':  # will not get accessed - PassthroughTransformer should be used instead
        if hasattr(column_transformer_get, '_df_columns'):
            if ((not isinstance(columns_get, slice))
                    and all(isinstance(col, str) for col in columns_get)):
                return columns_get
            else:
                return column_transformer_get._df_columns[columns_get]
        else:
            indices = np.arange(column_transformer_get._n_features)
            return ['x%d' % i for i in indices[columns_get]]
    if not hasattr(trans_get, 'get_feature_names_out'):
        # >>> Change: Return input column names if no method avaiable
        # Turn error into a warning
        # warnings.warn("Transformer %s (type %s) does not "
        #               "provide get_feature_names. "
        #               "Will return input column names if available"
        #               % (str(name), type(trans).__name__))
        # For transformers without a get_features_names method, use the input
        # names to the column transformer
        if columns_get is None:
            return []
        else:
            return [f for f in columns_get]

    return [f for f in trans_get.get_feature_names_out(columns_get)]


def get_feature_names_out(column_transformer):
    """A bit modified code from https://johaupt.github.io/blog/columnTransformer_feature_names.html
        get_feature_names is deprecated in 1.0 and will be removed
        in 1.2. Please use get_feature_names_out instead.
    Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, columns, _ in l_transformers:
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names_out(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [f for f in columns]
            feature_names.extend(_names)
        elif type(trans) == sklearn.compose.ColumnTransformer:
            # Recursive call on ColumnTransformer
            _names = get_feature_names_out(trans)
            # if ColumnTransformer has no transformer that returns names
            if len(_names) == 0:
                _names = [f for f in columns]
            feature_names.extend(_names)
        else:
            if columns is not None:  # a transformer in ColumnTransformer
                feature_names.extend(get_names(column_transformer, trans, columns))
            else:  # no longer in a ColumnTransformer nor Pipeline object
                feature_names = get_names(column_transformer, trans, feature_names)
    return feature_names
