"""
simulate_national_advanced.py

2023年で学習し、2024年（未知データ）で最終評価する。
パラメータは simulate_national.py のバリデーション結果から設定すること。
ここでは新たなパラメータ探索をしない（テスト汚染の防止）。
"""
import pandas as pd
import lightgbm as lgb
import warnings

from utils import add_features, FEATURES, bootstrap_roi

warnings.filterwarnings('ignore')

print("=== 🧪 全国版AI 未来シミュレーション（固定パラメータ・正規評価版）===")

# simulate_national.py の出力から設定する
BEST_THRESHOLD = 0.18
BEST_N2        = 2
BEST_N3        = 3

TRAIN_CSV = "race_results_20230101_to_20231231_ALL.csv"
TEST_CSV  = "race_results_20240101_to_20241231_ALL.csv"

try:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
except FileNotFoundError as e:
    print(f"エラー: {e}")
    exit()


def process(df):
    df = df.dropna(subset=['national_win_rate', 'motor_2ren', 'exhibition_time', 'tilt'])
    df = add_features(df)
    df = df.dropna(subset=['diff_win_rate', 'diff_motor', 'diff_exhibition'])
    df['target'] = df['rank'].apply(lambda x: int(x) - 1 if pd.notna(x) and int(x) <= 3 else 3)
    return df


train_df = process(train_df)
test_df  = process(test_df)

print(f"学習データ: {len(train_df)}行 (2023年)")
print(f"評価データ: {len(test_df)}行 (2024年)")

print(f"\n🧠 AIを学習中...")
model = lgb.LGBMClassifier(
    n_estimators=150, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, objective='multiclass',
    class_weight='balanced',
)
model.fit(train_df[FEATURES], train_df['target'], categorical_feature=['stadium', 'boat_num'])

print(f"🔮 2024年データを予測中...")
probs = model.predict_proba(test_df[FEATURES])
test_df = test_df.copy()
test_df['prob_1st'] = probs[:, 0]
test_df['prob_2nd'] = probs[:, 1]
test_df['prob_3rd'] = probs[:, 2]

# ── 全体評価 ───────────────────────────────────────────────────
print(f"\n=== 📊 固定パラメータで評価（穴閾値={BEST_THRESHOLD*100:.0f}%, 2着={BEST_N2}艇, 3着={BEST_N3}艇）===")

trades_all  = []
std_records = {}  # {std_code: [(inv, ret), ...]}

for (date, std, rno), race in test_df.groupby(['date', 'stadium', 'race_no']):
    if len(race) != 6:
        continue
    boat1_prob = race[race['boat_num'] == 1]['prob_1st'].values[0]
    if boat1_prob >= 0.50:
        continue

    non1 = race[race['boat_num'] != 1]
    best = non1.loc[non1['prob_1st'].idxmax()]
    if best['prob_1st'] < BEST_THRESHOLD:
        continue

    pivot  = int(best['boat_num'])
    remain = race[race['boat_num'] != pivot]

    boats_2nd = remain.nlargest(BEST_N2, 'prob_2nd')['boat_num'].astype(int).tolist()
    boats_3rd = remain.nlargest(BEST_N3, 'prob_3rd')['boat_num'].astype(int).tolist()

    tickets = [(pivot, b2, b3) for b2 in boats_2nd for b3 in boats_3rd if b2 != b3]
    if not tickets:
        continue

    inv = len(tickets) * 100
    a1  = race[race['rank'] == 1]['boat_num'].values
    a2  = race[race['rank'] == 2]['boat_num'].values
    a3  = race[race['rank'] == 3]['boat_num'].values

    ret = 0
    if len(a1) > 0 and len(a2) > 0 and len(a3) > 0:
        result = (int(a1[0]), int(a2[0]), int(a3[0]))
        if result in tickets:
            pays = race['payoff'].dropna()
            if not pays.empty:
                ret = pays.iloc[0]

    trades_all.append((inv, ret))

    std_code = f"{int(std):02d}"
    std_records.setdefault(std_code, []).append((inv, ret))

if not trades_all:
    print("条件に合うレースがありませんでした。")
    exit()

total_inv  = sum(t[0] for t in trades_all)
total_ret  = sum(t[1] for t in trades_all)
hit_count  = sum(1 for t in trades_all if t[1] > 0)
buy_count  = len(trades_all)
roi_point  = total_ret / total_inv * 100

ci_low, ci_mid, ci_high = bootstrap_roi(trades_all)

print(f"\n  賭けたレース数 : {buy_count:,}")
print(f"  的中率         : {hit_count/buy_count*100:.1f}%")
print(f"  収支           : {total_ret-total_inv:+,}円")
print(f"  ROI (点推定)   : {roi_point:.1f}%")
print(f"  ROI 95%CI      : {ci_low:.1f}% 〜 {ci_high:.1f}%  (中央値: {ci_mid:.1f}%)")

if ci_low > 100:
    print("\n✅ CI下限も100%超。この戦略は統計的に有意なエッジがある可能性が高い。")
elif ci_high > 100:
    print("\n⚠️ エッジがある可能性はあるが、CI下限が100%未満。2025年データで再確認推奨。")
else:
    print("\n❌ 2024年通年データではエッジが確認できません。モデル改良が必要です。")

# ── 会場別分析 ─────────────────────────────────────────────────
print(f"\n=== 🏟️ 会場別成績 ===")

std_rows = []
for std_code, records in std_records.items():
    if not records:
        continue
    inv = sum(r[0] for r in records)
    ret = sum(r[1] for r in records)
    hits = sum(1 for r in records if r[1] > 0)
    cnt  = len(records)
    ci   = bootstrap_roi(records)
    std_rows.append({
        "会場": std_code,
        "R数":  cnt,
        "的中率": f"{hits/cnt*100:.0f}%",
        "ROI":    ret / inv * 100,
        "収支":   ret - inv,
        "CI下限": ci[0],
        "CI上限": ci[2],
    })

std_df = pd.DataFrame(std_rows).sort_values('ROI', ascending=False)

# CI下限が100%を超える会場（統計的に有望）
reliable = std_df[std_df['CI下限'] > 100]
print(f"\n✅ 【統計的に有望な会場（CI下限>100%）】 → predict_national.py の GOOD_STADIUMS に設定")
if reliable.empty:
    print("  該当なし（サンプル不足、または戦略の見直しが必要）")
else:
    for _, r in reliable.iterrows():
        print(
            f"  会場 {r['会場']} : {r['R数']:3.0f}R | "
            f"ROI {r['ROI']:6.1f}% | "
            f"95%CI [{r['CI下限']:.0f}%〜{r['CI上限']:.0f}%] | "
            f"収支 {r['収支']:+8.0f}円"
        )

print(f"\n❌ 【成績ワースト5会場】")
for _, r in std_df.tail(5).iterrows():
    print(
        f"  会場 {r['会場']} : {r['R数']:3.0f}R | "
        f"ROI {r['ROI']:6.1f}% | "
        f"95%CI [{r['CI下限']:.0f}%〜{r['CI上限']:.0f}%] | "
        f"収支 {r['収支']:+8.0f}円"
    )

print(f"\n全会場一覧:")
print(std_df[['会場', 'R数', '的中率', 'ROI', 'CI下限', 'CI上限', '収支']].to_string(index=False))
