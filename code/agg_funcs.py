from collections import Counter

def mode(x):
    return Counter(x).most_common(1)[0][0]

def num_unique(x):
    return len(set(x))

def flatten_column(df):
    df.columns = ["__".join(col).strip() for col in df.columns.values]
    return df

def quantile25(series):
   return series.quantile(0.25)

def quantile75(series):
   return series.quantile(0.75)

def aggregation(x_train, x_test, agg_base_col, agg_dic):
    agg_train = x_train.groupby(agg_base_col).agg(agg_dic)
    agg_train = flatten_column(agg_train)
    agg_test = x_test.groupby(agg_base_col).agg(agg_dic)
    agg_test = flatten_column(agg_test)
    return agg_train, agg_test