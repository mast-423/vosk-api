import numpy as np
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


def plackett_luce_prob(scores: dict, combo: tuple) -> float:
    """
    Plackett-Luce モデルで特定3連単の確率を計算する。
    scores: {boat_num: strength}  (モデルの prob_1st を強度として使用)
    combo:  (1着艇番, 2着艇番, 3着艇番)
    """
    a, b, c = combo
    vals = {k: max(float(v), 1e-12) for k, v in scores.items()}
    total = sum(vals.values())
    if total == 0:
        return 0.0
    p_a = vals.get(a, 0.0) / total
    rem1 = total - vals.get(a, 0.0)
    p_b = vals.get(b, 0.0) / rem1 if rem1 > 0 else 0.0
    rem2 = rem1 - vals.get(b, 0.0)
    p_c = vals.get(c, 0.0) / rem2 if rem2 > 0 else 0.0
    return p_a * p_b * p_c


def kelly_fraction(prob: float, payout: int) -> float:
    """
    Full Kelly 比率を返す。
    payout: 100円賭けたときの払戻金（例: 4200 → 41倍のネット収益）
    マイナス期待値の場合は 0.0 を返す。
    """
    if payout <= 100:
        return 0.0
    b = payout / 100.0 - 1.0  # ネット収益倍率
    ev = b * prob - (1.0 - prob)
    if ev <= 0:
        return 0.0
    return ev / b


def bootstrap_roi(trades: list, n_boot: int = 2000, seed: int = 42) -> tuple:
    """
    trades: [(investment, payout), ...] のリスト
    ROI の 95% 信頼区間を (下限, 中央値, 上限) で返す。
    """
    rng = np.random.default_rng(seed)
    n = len(trades)
    if n == 0:
        return (0.0, 0.0, 0.0)
    arr = np.array(trades, dtype=float)
    rois = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = arr[idx]
        inv = sample[:, 0].sum()
        ret = sample[:, 1].sum()
        rois.append(ret / inv * 100 if inv > 0 else 0.0)
    return tuple(np.percentile(rois, [2.5, 50, 97.5]))
