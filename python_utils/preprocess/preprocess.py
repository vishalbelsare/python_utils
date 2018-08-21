import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype


def time_discretise_panel_data(df, panelvar=None, timevar=None, freq=None,
                               aggs=None):
    """
    Discretises time variable of panel data-frame (df) by aggregating
    variables into equal time-bins for each panel.


    :param df: pandas data-frame
    :param panelvar: panel identifier
    :param timevar: time variable of panel data
    :param freq: frequency for time-binning
    :param aggs: aggregation  as in pandas agg function
    :return: time-discretised aggregated data-frame
    """

    df_grouped = (df
                  .groupby([pd.Grouper(key=timevar, freq=freq), panelvar])
                  .agg(aggs)
                  .reset_index())

    unique_panels = df_grouped[panelvar].unique()
    start = df[timevar].min()
    end = df[timevar].max()
    time_range = pd.date_range(start=start, end=end, freq=freq)
    full_index = pd.MultiIndex.from_product([unique_panels, time_range],
                                            names=[panelvar, timevar])

    df_binned = (df_grouped
                 .set_index([panelvar, timevar])
                 .reindex(full_index))
    return df_binned


def collapse_multicolumn_index(df):
    """
    Transform in-place a 2-level column multi-index of a pandas data-frame
    into a simple 1-level index by combining column names from both levels
    :param df: pandas data-frame
    """

    colnames = df.columns.get_level_values(0) + '_' \
                                              + df.columns.get_level_values(1)
    df.columns = df.columns.droplevel()
    df.columns = colnames


def str_dtype_to_cats(df):
    """
    Makes in-place transformations of all string_dtypes in data-frame (df)
    to ordered categories
    """

    for col_name, col_series in df.items():
        if is_string_dtype(col_series):
            df[col_name] = col_series.astype('category').cat.as_ordered()


def clean_str_dtype(df, value='_'):
    """
    Makes in-place transformations of all string_dtypes columns in data-frame
    (df) to lower-case string values without white spaces, by default
    replacing white spaces with underscores, otherwise replacing them with
    'value'
    """

    for col_name, col_series in df.items():
        if is_string_dtype(col_series):
            df[col_name] = (col_series.str.strip()
                                      .str.replace(' ', value)
                                      .str.lower())


def clean_col_names(df):
    """
    Makes in-place transformations of all column names in data-frame (df) to
    lower-case names separated with underscores without white spaces
    """

    df.columns = [col.strip().replace(' ', '_').replace('.', '_').lower() 
                  for col in df.columns]


def add_date_parts(df, col_name, suffix='', drop=False):
    """
    Extracts date properties from a column (col_name) in data-frame (df)
    and adds them as new features in-place.


    Parameters:
    -----------
    df: A pandas data frame, df gains several new columns
    col_name: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with
        pd.to_datetime.
    drop: If true then the original date column will be removed.
    suffix: A string that is used as a suffix for the added features
    """

    col = df[col_name]
    if not np.issubdtype(col.dtype, np.datetime64):
        df[col_name] = col = pd.to_datetime(col, infer_datetime_format=True)

    for date_part in ('year', 'quarter', 'month', 'week', 'day',
                      'dayofweek', 'dayofyear', 'is_month_end',
                      'is_month_start', 'is_quarter_end',
                      'is_quarter_start', 'is_year_end',
                      'is_year_start'):
        df[suffix + date_part] = getattr(col.dt, date_part)

    if drop:
        df.drop(col_name, axis=1, inplace=True)


def add_time_parts(df, col_name, suffix='', drop=False):
    """Extracts time properties from a column (col_name) in data-frame (df)
    and adds them as new features in-place.

    Parameters:
    -----------
    df: A pandas data frame, df gains several new columns
    col_name: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with
        pd.to_datetime.
    drop: If true then the original date column will be removed.
    suffix: A string that is used as a suffix for the added features
    """

    col = df[col_name]
    if not np.issubdtype(col.dtype, np.datetime64):
        df[col_name] = col = pd.to_datetime(col, infer_datetime_format=True)

    for date_part in ('hour', 'minute', 'second', 'microsecond', 'nanosecond'):
        df[suffix + date_part] = getattr(col.dt, date_part)

    df[suffix + 'elapsed'] = col.astype(np.int64) // 10**9

    if drop:
        df.drop(col_name, axis=1, inplace=True)
