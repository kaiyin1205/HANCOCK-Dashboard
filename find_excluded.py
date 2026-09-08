import pandas as pd

targets = pd.read_csv('features/targets.csv', dtype={'patient_id': str})
split   = pd.read_json('results/dataset_split_in.json', dtype={'patient_id': str})[['patient_id', 'dataset']]
df      = split.merge(targets, on='patient_id', how='inner')

kept = df[
    ((df.recurrence == 'yes') & (df.days_to_recurrence <= 365*3)) |
    ((df.recurrence == 'no') & ((df.days_to_last_information > 365*3) | (df.survival_status == 'living')))
]

excluded = df[~df.patient_id.isin(kept.patient_id)]
print(f'Total patients: {len(df)}')
print(f'Kept: {len(kept)}')
print(f'Excluded: {len(excluded)}')
print(f'Example excluded IDs: {excluded.patient_id.tolist()[:5]}')
print(excluded[['patient_id', 'recurrence', 'days_to_recurrence', 'days_to_last_information', 'survival_status']].head())