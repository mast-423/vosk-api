import pandas as pd
import lightgbm as lgb
import warnings

from utils import add_features, FEATURES

warnings.filterwarnings('ignore')

print("=== 🧪 全国版AI タイムシリーズ・シミュレーション ===")

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

train_df = df[df['date'] < 20230901]
test_df  = df[df['date'] >= 20230901]

print(f"\n🧠 学習中... (学習: {len(train_df)}行 / 1〜8月)")
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, objective='multiclass')
model.fit(train_df[FEATURES], train_df['target'], categorical_feature=['stadium', 'boat_num'])

print(f"🔮 予測中... (テスト: {len(test_df)}行 / 9〜12月)")
probs = model.predict_proba(test_df[FEATURES])
test_df = test_df.copy()
test_df['prob_1st'] = probs[:, 0]
test_df['prob_2nd'] = probs[:, 1]
test_df['prob_3rd'] = probs[:, 2]

print("\n=== 💰 最強設定の探索中... ===")

thresholds_1st = [0.10, 0.12, 0.15, 0.18]
num_2nd_list   = [2, 3, 4]
num_3rd_list   = [3, 4, 5]

results_log    = []
grouped_test   = test_df.groupby(['date', 'stadium', 'race_no'])

for t1 in thresholds_1st:
    for n2 in num_2nd_list:
        for n3 in num_3rd_list:
            if n2 > n3:
                continue

            total_inv = total_ret = buy_cnt = hit_cnt = total_tickets = 0

            for (date, std, rno), race_data in grouped_test:
                if len(race_data) != 6:
                    continue

                boat1_prob = race_data[race_data['boat_num'] == 1]['prob_1st'].values[0]
                if boat1_prob >= 0.50:
                    continue

                non1     = race_data[race_data['boat_num'] != 1]
                boat_1st = non1.loc[non1['prob_1st'].idxmax()]
                if boat_1st['prob_1st'] < t1:
                    continue

                pivot  = int(boat_1st['boat_num'])
                remain = race_data[race_data['boat_num'] != pivot]

                boats_2nd = remain.nlargest(n2, 'prob_2nd')['boat_num'].astype(int).tolist()
                boats_3rd = remain.nlargest(n3, 'prob_3rd')['boat_num'].astype(int).tolist()

                tickets = [
                    (pivot, b2, b3)
                    for b2 in boats_2nd
                    for b3 in boats_3rd
                    if b2 != b3
                ]
                if not tickets:
                    continue

                buy_cnt       += 1
                total_tickets += len(tickets)
                total_inv     += len(tickets) * 100

                a1 = race_data[race_data['rank'] == 1]['boat_num'].values
                a2 = race_data[race_data['rank'] == 2]['boat_num'].values
                a3 = race_data[race_data['rank'] == 3]['boat_num'].values

                if len(a1) > 0 and len(a2) > 0 and len(a3) > 0:
                    result = (int(a1[0]), int(a2[0]), int(a3[0]))
                    if result in tickets:
                        hit_cnt += 1
                        pay = race_data['payoff'].dropna()
                        if not pay.empty:
                            total_ret += pay.iloc[0]

            if buy_cnt > 0:
                avg_tickets = total_tickets / buy_cnt
                results_log.append({
                    "設定":       f"穴勝率={t1*100:.0f}%, 2着={n2}艇, 3着={n3}艇",
                    "平均点数":   f"{avg_tickets:.1f}点",
                    "レース数":   buy_cnt,
                    "的中率":     f"{hit_cnt/buy_cnt*100:.1f}%",
                    "ROI":        f"{total_ret/total_inv*100:.1f}%",
                    "収支":       total_ret - total_inv,
                })

print("\n=== 🔮 未知データ（23年9〜12月）トップ10 ===")
results_df = pd.DataFrame(results_log)
if not results_df.empty:
    results_df['ROI_num'] = results_df['ROI'].str.replace('%', '').astype(float)
    print(results_df.sort_values('ROI_num', ascending=False).drop(columns='ROI_num').head(10).to_string(index=False))
else:
    print("条件に合うレースがありませんでした。")
