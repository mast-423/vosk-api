import pandas as pd

FEATURES = [
    'stadium', 'boat_num', 'national_win_rate', 'motor_2ren',
    'wind_speed', 'wave_height', 'exhibition_time', 'tilt',
    'diff_win_rate', 'diff_motor', 'diff_exhibition',
    'exhibition_rank', 'win_rate_rank',
]


def add_features(df):
    boat1 = df[df['boat_num'] == 1][
        ['date', 'stadium', 'race_no', 'national_win_rate', 'motor_2ren', 'exhibition_time']
    ].rename(columns={
        'national_win_rate': 'boat1_win_rate',
        'motor_2ren':        'boat1_motor',
        'exhibition_time':   'boat1_exhibition',
    })
    out = pd.merge(df, boat1, on=['date', 'stadium', 'race_no'], how='left')
    out['diff_win_rate']   = out['national_win_rate'] - out['boat1_win_rate']
    out['diff_motor']      = out['motor_2ren']        - out['boat1_motor']
    out['diff_exhibition'] = out['exhibition_time']   - out['boat1_exhibition']
    g = out.groupby(['date', 'stadium', 'race_no'])
    out['exhibition_rank'] = g['exhibition_time'].rank(method='min')
    out['win_rate_rank']   = g['national_win_rate'].rank(method='min', ascending=False)
    return out
