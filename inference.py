import pandas as pd
import catboost as cb



def fill_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in dataframe with inferred values
    """

    # columns that have missing values are inference_columns
    inference_columns = list()
    
    # columns that are complete are used as features
    feature_columns = list()

    for c in df.columns:
        n_missing = df[c].isnull().sum()
        if n_missing > 0:
            inference_columns.append([n_missing, c])
        else:
            feature_columns.append(c)
    # sort by number of missing values
    sorted(inference_columns, key=lambda x: x[0])

    for n, c in inference_columns:    
        # print(f"{c}: {n} missing values")
        df[c] = _infer_columns(df, c, feature_columns)
        feature_columns.append(c)

    return df.fillna(0)


def _infer_columns(df: pd.DataFrame, infer_col_name:str, feature_col_names:list) -> pd.DataFrame:
    """
        Infer missing values in a column using columns in feature_col_names
    """
    # dummy implementation - fill with most frequent value
    infer_col = df[infer_col_name].copy(deep=True)
    mode_val = infer_col.mode()
    # print(mode_val[0])
    if not( pd.isnull(mode_val[0]) or pd.isna(mode_val[0])):
        fill_val = mode_val[0]
    else:
        fill_val = 0
    infer_col = infer_col.fillna(fill_val)
    
    return infer_col


def test_fill_dataframe():

    for testfile in [
        "example_data/merged_data_cleaned.csv",
        "example_data/glaciers/glaciers.csv",
        "example_data/city_temperature.csv"
    ]:

        df = pd.read_csv(testfile)
        df = df.fillna(0)
        df_filled = fill_dataframe(df)
        assert df_filled.equals(df), "fill_dataframe should not change a filled dataframe"
        assert df_filled.isnull().sum().sum() == 0, "fill_dataframe should return a dataframe with no missing values"

        df = pd.read_csv(testfile)
        #print(f"null-values before: {df.isnull().sum().sum()}")
        df_filled = fill_dataframe(df)
        #print(f"null-values after: {df_filled.isnull().sum().sum()}")
        assert df_filled.isnull().sum().sum() == 0, f"{testfile} : fill_dataframe should return a dataframe with no missing values"
