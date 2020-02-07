import pandas as pd
import numpy as np
        
# decide subplot shape
def row_col(n):
    """
    n is the number of features
    """
    if n <= 3:
        ncol = n
        nrow = 1
    else:
        ncol = 3
        nrow = np.ceil(n / ncol)
    return int(nrow), int(ncol)

# reduce memory
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)                    
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

# sensitivity index
def d_prime(train, test, feature):
    tr = train[feature].values
    ts = test[feature].values
    return (np.nanmean(tr) - np.nanmean(ts)) / np.sqrt(0.5 * (np.nanvar(tr) + np.nanvar(ts)))

import numpy as np
import os
import datetime
import glob
import shutil

def prepare_result_folder(result_dir_original, version_name, comment):
    ## result dir setting
#     result_dir_base, result_dir, now = prepare_result_folder(result_dir_original = "results", version_name = version_name, comment = comment)

    ## save result
#     elapsed_time = time.time() - t_start
#     with open(result_dir_base + "/result.txt", "a") as f:
#         f.write("{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\n".format(params["model_type"], auc, logloss, cal, brier, mse_cal, r2_cal, elapsed_time, now, comment))
#     pd.to_pickle(params, result_dir + "/params.pkl")
           
    # 結果を保存するdirを用意する．
    # 同じdirに存在する.pyを全部コピーする
    result_dir_base = f"{result_dir_original}_{version_name}"
    os.makedirs(result_dir_base, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fn = comment + "_" + now
    result_dir = result_dir_base + "/" + fn
    os.makedirs(result_dir, exist_ok = True)
    os.makedirs(result_dir + "/script", exist_ok = True)
    copy_pys = glob.glob("./*py")
    for py in copy_pys:
        shutil.copy(py, result_dir + "/script/")
    print("result_dir : ", result_dir)
    return result_dir_base, result_dir, now
