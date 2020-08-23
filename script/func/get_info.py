def get_info(target_df, topN=10, zero=False, u_val=False):
    max_row = len(target_df)
    print(f'Shape: {target_df.shape}')
    
    df = target_df.dtypes.to_frame()
    df.columns = ['DataType']
    df['Nulls'] = target_df.isnull().sum()
    df['Null%'] = df['Nulls'] / max_row * 100
    df['Uniques'] = target_df.nunique()
    df['Unique%'] = df['Uniques'] / max_row * 100
    
    if zero:
        df['Zeros'] = (target_df == 0).sum()
        df['Zero%'] = df['Zeros'] / max_row
    
    # stats
    df['Min']   = target_df.min(numeric_only=True)
    df['Mean']  = target_df.mean(numeric_only=True)
    df['Max']   = target_df.max(numeric_only=True)
    df['Std']   = target_df.std(numeric_only=True)
    
    # top 10 values
    df[f'top{topN} val'] = 0
    df[f'top{topN} cnt'] = 0
    df[f'top{topN} raito'] = 0
    for c in df.index:
        vc = target_df[c].value_counts().head(topN)
        val = list(vc.index)
        cnt = list(vc.values)
        raito = list((vc.values / max_row).round(2))
        df.loc[c, f'top{topN} val'] = str(val)
        df.loc[c, f'top{topN} cnt'] = str(cnt)
        df.loc[c, f'top{topN} raito'] = str(raito)
        
    if u_val:
        df['u_val'] = [target_df[col].unique() for col in cols]
        
    return df