"""
simulate_national.py

正しい評価フロー:
  1. Train:      Jan〜Jun 2023  (モデル学習)
  2. Validation: Jul〜Aug 2023  (ハイパーパラメータ選択)
  3. Test:       Sep〜Dec 2023  (1回だけ触れる最終評価)

旧コードの問題: テストデータでグリッドサーチして最良値を報告 → 実質過学習
修正: バリデーションでパラメータを固定してからテストで評価
"""
import random
import pandas as pd
import lightgbm as lgb
import warnings

from utils import add_features, FEATURES, bootstrap_roi

warnings.filterwarnings('ignore')

print("=== 🧪 全国版AI タイムシリーズ・シミュレーション（正規評価版）===")

CSV_FILENAME = "race_results_20230101_to_20231231_ALL.csv"
try:
    df = pd.read_csv(CSV_FILENAME)
except FileNotFoundError:
    print(f"エラー: {CSV_FILENAME} が見つかりません。")
    exit()

df = df.dropna(subset=['national_win_rate', 'motor_2ren', 'exhibition_time', 'tilt'])
df = add_features(df)
df = df.dropna(subset=['diff_win_rate', 'diff_motor', 'diff_exhibition'])
df['target'] = df['rank'].apply(lambda x: int(x) - 1 if pd.notna(x) and int(x) <= 3 else 3)

# ── データ分割 ─────────────────────────────────────────────────
train_df = df[df['date'] < 20230701]   # Jan〜Jun
val_df   = df[(df['date'] >= 20230701) & (df['date'] < 20230901)]  # Jul〜Aug
test_df  = df[df['date'] >= 20230901]  # Sep〜Dec（最後に1回だけ使う）

print(f"\n学習: {len(train_df)}行 (1〜6月)")
print(f"検証: {len(val_df)}行 (7〜8月)")
print(f"評価: {len(test_df)}行 (9〜12月)\n")

print("🧠 AIを学習中...")
model = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.05, random_state=42,
    objective='multiclass', class_weight='balanced',
)
model.fit(train_df[FEATURES], train_df['target'], categorical_feature=['stadium', 'boat_num'])


def run_simulation(model, eval_df, t1, n2, n3):
    """指定パラメータでシミュレーション実行。trades リストを返す。"""
    probs = model.predict_proba(eval_df[FEATURES])
    df2 = eval_df.copy()
    df2['prob_1st'] = probs[:, 0]
    df2['prob_2nd'] = probs[:, 1]
    df2['prob_3rd'] = probs[:, 2]

    trades = []  # [(投資額, 回収額), ...]

    for (date, std, rno), race in df2.groupby(['date', 'stadium', 'race_no']):
        if len(race) != 6:
            continue
        boat1_prob = race[race['boat_num'] == 1]['prob_1st'].values[0]
        if boat1_prob >= 0.50:
            continue

        non1     = race[race['boat_num'] != 1]
        best     = non1.loc[non1['prob_1st'].idxmax()]
        if best['prob_1st'] < t1:
            continue

        pivot  = int(best['boat_num'])
        remain = race[race['boat_num'] != pivot]

        boats_2nd = remain.nlargest(n2, 'prob_2nd')['boat_num'].astype(int).tolist()
        boats_3rd = remain.nlargest(n3, 'prob_3rd')['boat_num'].astype(int).tolist()

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
        trades.append((inv, ret))
    return trades


# ── Phase 1: バリデーションでパラメータ選択 ───────────────────
print("🔍 バリデーションセット（7〜8月）でパラメータ選択中...")

thresholds = [0.10, 0.12, 0.15, 0.18, 0.20]
n2_list    = [2, 3, 4]
n3_list    = [3, 4, 5]

best_roi_val = -float('inf')
best_params  = None

for t1 in thresholds:
    for n2 in n2_list:
        for n3 in n3_list:
            if n2 > n3:
                continue
            trades = run_simulation(model, val_df, t1, n2, n3)
            if not trades:
                continue
            inv = sum(t[0] for t in trades)
            ret = sum(t[1] for t in trades)
            roi = ret / inv * 100 if inv > 0 else 0
            if roi > best_roi_val:
                best_roi_val = roi
                best_params  = (t1, n2, n3)

if best_params is None:
    print("バリデーションで条件に合うレースがありませんでした。")
    exit()

t1_best, n2_best, n3_best = best_params
print(f"✅ バリデーション最良パラメータ: 穴閾値={t1_best*100:.0f}%, 2着={n2_best}艇, 3着={n3_best}艇 (Val ROI: {best_roi_val:.1f}%)\n")

# ── Phase 2: テストセットで最終評価（1回のみ） ───────────────
print("📊 テストセット（9〜12月）で最終評価...")
trades_test = run_simulation(model, test_df, t1_best, n2_best, n3_best)

if not trades_test:
    print("テストデータで条件に合うレースがありませんでした。")
    exit()

total_inv  = sum(t[0] for t in trades_test)
total_ret  = sum(t[1] for t in trades_test)
hit_count  = sum(1 for t in trades_test if t[1] > 0)
buy_count  = len(trades_test)
roi_point  = total_ret / total_inv * 100

ci_low, ci_mid, ci_high = bootstrap_roi(trades_test)

print(f"\n=== 🔮 最終評価結果（9〜12月 / 未知データ）===")
print(f"  賭けたレース数 : {buy_count:,}")
print(f"  的中数         : {hit_count:,}")
print(f"  的中率         : {hit_count/buy_count*100:.1f}%")
print(f"  総投資         : {total_inv:,}円")
print(f"  総回収         : {total_ret:,}円")
print(f"  収支           : {total_ret-total_inv:+,}円")
print(f"  ROI (点推定)   : {roi_point:.1f}%")
print(f"  ROI 95%CI      : {ci_low:.1f}% 〜 {ci_high:.1f}%  (中央値: {ci_mid:.1f}%)")
print()

if ci_low > 100:
    print("✅ 信頼区間の下限も100%を超えています。有望な戦略と判断できます。")
elif ci_high > 100:
    print("⚠️ 点推定はプラスですが、CI下限が100%を下回ります。サンプル不足の可能性があります。")
else:
    print("❌ 信頼区間の上限も100%を下回っています。戦略の有効性が確認できません。")

print(f"\n推奨パラメータ（simulate_national_advanced.py に設定してください）:")
print(f"  BEST_THRESHOLD = {t1_best}")
print(f"  BEST_N2        = {n2_best}")
print(f"  BEST_N3        = {n3_best}")
