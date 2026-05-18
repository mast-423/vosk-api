import pandas as pd
import lightgbm as lgb
import warnings

from utils import add_features, FEATURES

warnings.filterwarnings('ignore')

print("=== 🧪 全国版AI 未来シミュレーション (リミッター解除版) ===")

TRAIN_CSV = "race_results_20230101_to_20231231_ALL.csv"
TEST_CSV  = "race_results_20240303_to_20240331_ALL.csv"

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


print("AIを構築中...")
train_df = process(train_df)
test_df  = process(test_df)

print(f"\n🧠 学習中... ({len(train_df)}行)")
model = lgb.LGBMClassifier(
    n_estimators=150, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, objective='multiclass',
)
model.fit(train_df[FEATURES], train_df['target'], categorical_feature=['stadium', 'boat_num'])

print(f"🔮 予測中... ({len(test_df)}行)")
probs = model.predict_proba(test_df[FEATURES])
test_df = test_df.copy()
test_df['prob_1st'] = probs[:, 0]
test_df['prob_2nd'] = probs[:, 1]
test_df['prob_3rd'] = probs[:, 2]

print("\n=== 💰 2024年データで最強設定を探索中... ===")

thresholds_1st = [0.15, 0.18, 0.20]
num_2nd_list   = [3, 4]
num_3rd_list   = [4, 5]

results_log  = []
grouped_test = test_df.groupby(['date', 'stadium', 'race_no'])

for t1 in thresholds_1st:
    for n2 in num_2nd_list:
        for n3 in num_3rd_list:
            if n2 > n3:
                continue

            total_inv = total_ret = buy_cnt = hit_cnt = total_tickets = 0
            std_stats = {f"{i:02d}": {'inv': 0, 'ret': 0, 'hit': 0, 'buy': 0} for i in range(1, 25)}

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

                inv = len(tickets) * 100
                buy_cnt       += 1
                total_tickets += len(tickets)
                total_inv     += inv

                std_code = f"{int(std):02d}"
                std_stats[std_code]['buy'] += 1
                std_stats[std_code]['inv'] += inv

                a1 = race_data[race_data['rank'] == 1]['boat_num'].values
                a2 = race_data[race_data['rank'] == 2]['boat_num'].values
                a3 = race_data[race_data['rank'] == 3]['boat_num'].values

                if len(a1) > 0 and len(a2) > 0 and len(a3) > 0:
                    result = (int(a1[0]), int(a2[0]), int(a3[0]))
                    if result in tickets:
                        hit_cnt += 1
                        std_stats[std_code]['hit'] += 1
                        pay = race_data['payoff'].dropna()
                        if not pay.empty:
                            p = pay.iloc[0]
                            total_ret += p
                            std_stats[std_code]['ret'] += p

            if buy_cnt > 0:
                avg_tickets = total_tickets / buy_cnt
                results_log.append({
                    "設定":     f"穴勝率={t1*100:.0f}%, 2着={n2}艇, 3着={n3}艇",
                    "平均点数": f"{avg_tickets:.1f}点",
                    "レース数": buy_cnt,
                    "的中率":   f"{hit_cnt/buy_cnt*100:.1f}%",
                    "ROI":      f"{total_ret/total_inv*100:.1f}%",
                    "収支":     total_ret - total_inv,
                    "_std":     std_stats,
                })

print("\n=== 🔮 未来テスト（24年3月）全体トップ5 ===")
results_df = pd.DataFrame(results_log)

if results_df.empty:
    print("条件に合うレースがありませんでした。")
else:
    results_df['ROI_num'] = results_df['ROI'].str.replace('%', '').astype(float)
    top5 = results_df.sort_values('ROI_num', ascending=False).head(5)
    print(top5.drop(columns=['ROI_num', '_std']).to_string(index=False))

    best = top5.iloc[0]
    print(f"\n=== 🏟️ 最強設定の【会場別】成績 ===")
    print(f"対象設定: {best['設定']}")

    std_log = []
    for code, stats in best['_std'].items():
        if stats['inv'] > 0:
            std_log.append({
                "会場":     code,
                "レース数": stats['buy'],
                "ROI":      (stats['ret'] / stats['inv']) * 100,
                "収支":     stats['ret'] - stats['inv'],
            })

    std_df = pd.DataFrame(std_log).sort_values('ROI', ascending=False)

    print("\n✅ 【AIが得意な会場 トップ5】")
    for _, r in std_df.head(5).iterrows():
        print(f"会場 {r['会場']} : {r['レース数']:2.0f}R | ROI {r['ROI']:6.1f}% | 収支 {r['収支']:+8.0f}円")

    print("\n❌ 【AIが苦手な会場 ワースト5】")
    for _, r in std_df.tail(5).iterrows():
        print(f"会場 {r['会場']} : {r['レース数']:2.0f}R | ROI {r['ROI']:6.1f}% | 収支 {r['収支']:+8.0f}円")
