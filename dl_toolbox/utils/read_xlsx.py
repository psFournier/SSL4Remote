import pandas as pd

paths = [
    '/d/pfournie/ai4geo/outputs/version_50/checkpoints/metrics_fold4.xlsx',
    '/d/pfournie/ai4geo/outputs/version_50/checkpoints/metrics.xlsx'
]

df_list = []
for path in paths:

    df_list.append(pd.read_excel(path, index_col=0, sheet_name='average_metrics'))

result = pd.concat(df_list, ignore_index=True)
print(result)
