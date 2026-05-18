import requests
from bs4 import BeautifulSoup
import pandas as pd
import lightgbm as lgb
import joblib
import unicodedata
import os
import warnings
from datetime import datetime

from utils import add_features, FEATURES

warnings.filterwarnings('ignore')

TARGET_DATE = datetime.today().strftime('%Y%m%d')
print(f"=== 🤖 全国対応版・穴党AI予想システム ({TARGET_DATE}) ===")

while True:
    try:
        jcd_num = int(input("🎯 会場コード（1〜24）: "))
        if 1 <= jcd_num <= 24:
            JCD = f"{jcd_num:02d}"
            break
        print("⚠️ 1から24の数字を入力してください。")
    except ValueError:
        print("⚠️ 正しい数字を入力してください。")

while True:
    try:
        TARGET_RACE = int(input("🎯 レース番号（1〜12）: "))
        if 1 <= TARGET_RACE <= 12:
            break
        print("⚠️ 1から12の数字を入力してください。")
    except ValueError:
        print("⚠️ 正しい数字を入力してください。")

print(f"\n🚀 【会場: {JCD} / {TARGET_RACE}R】 の予想を開始します...\n")

# ── 得意会場フィルター ─────────────────────────────────────────
# simulate_national_advanced.py 実行後に ROI が安定して高い会場を設定する
# 例: GOOD_STADIUMS = ["03", "07", "15"]
GOOD_STADIUMS = []

if GOOD_STADIUMS and JCD not in GOOD_STADIUMS:
    print(f"⚠️ 会場 {JCD} はAIの得意会場リスト外です。")
    ans = input("それでも予測を続けますか？ (y/n): ").strip().lower()
    if ans != 'y':
        exit()

THRESHOLD_1ST = 0.18  # ROI 239.4% の設定（2023年バックテスト結果）
NUM_2ND = 2
NUM_3RD = 3

# ── モデルの準備（CSVより新しくなければキャッシュ使用） ────────
CSV_FILENAME   = "race_results_20230101_to_20231231_ALL.csv"
MODEL_FILENAME = "boatrace_model.pkl"

retrain_needed = (
    not os.path.exists(MODEL_FILENAME)
    or (os.path.exists(CSV_FILENAME)
        and os.path.getmtime(CSV_FILENAME) > os.path.getmtime(MODEL_FILENAME))
)

if retrain_needed:
    print("全国24場データからAIを構築中...（数秒かかります）")
    df = pd.read_csv(CSV_FILENAME)
    df = df.dropna(subset=['national_win_rate', 'motor_2ren', 'exhibition_time', 'tilt'])
    df = add_features(df)
    df = df.dropna(subset=['diff_win_rate', 'diff_motor', 'diff_exhibition'])
    df['target'] = df['rank'].apply(lambda x: int(x) - 1 if pd.notna(x) and int(x) <= 3 else 3)

    model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.05, random_state=42, objective='multiclass'
    )
    model.fit(df[FEATURES], df['target'], categorical_feature=['stadium', 'boat_num'])
    joblib.dump(model, MODEL_FILENAME)
    print(f"モデルを {MODEL_FILENAME} に保存しました。")
else:
    print("保存済みモデルを読み込んでいます...")
    model = joblib.load(MODEL_FILENAME)

# ── 直前情報の取得 ────────────────────────────────────────────
print("公式サイトから直前情報を取得中...\n")


def get_live_data(date, jcd, rno):
    rows = []

    soup_list = BeautifulSoup(
        requests.get(
            f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={date}",
            timeout=10,
        ).content,
        'html.parser',
    )
    race_list_data = {}
    for tbody in soup_list.find_all("tbody", class_="is-fs12"):
        cols = tbody.find_all("td")
        if len(cols) >= 8:
            try:
                raw = unicodedata.normalize('NFKC', cols[0].get_text(strip=True))
                if not raw or not raw[0].isdigit():
                    continue
                b = int(raw[0])
                if b not in range(1, 7):
                    continue
                a_tag = cols[2].find('a')
                name = a_tag.get_text(strip=True).replace('　', '') if a_tag else "不明"
                rate_parts  = cols[5].get_text(separator=" ", strip=True).split()
                motor_parts = cols[7].get_text(separator=" ", strip=True).split()
                race_list_data[b] = {
                    "racer_name":       name,
                    "national_win_rate": rate_parts[0]  if rate_parts            else None,
                    "motor_2ren":        motor_parts[1] if len(motor_parts) > 1  else None,
                }
            except Exception:
                pass

    soup_before = BeautifulSoup(
        requests.get(
            f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={date}",
            timeout=10,
        ).content,
        'html.parser',
    )
    wind_speed, wave_height = 0.0, 0.0
    try:
        w = soup_before.find("div", class_="is-wind").find("span", class_="weather1_bodyUnitLabelData")
        if w: wind_speed  = float(w.get_text(strip=True).replace('m', ''))
        v = soup_before.find("div", class_="is-wave").find("span", class_="weather1_bodyUnitLabelData")
        if v: wave_height = float(v.get_text(strip=True).replace('cm', ''))
    except Exception:
        pass

    before_data = {}
    for tbody in soup_before.find_all("tbody"):
        for row in tbody.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 6:
                try:
                    b = int(unicodedata.normalize('NFKC', cols[0].get_text(strip=True)))
                    if b in range(1, 7):
                        before_data[b] = {
                            "exhibition_time": cols[4].get_text(strip=True),
                            "tilt":            cols[5].get_text(strip=True),
                        }
                except Exception:
                    pass

    for b in range(1, 7):
        if b in race_list_data and b in before_data:
            try:
                rows.append({
                    "date":     date,
                    "stadium":  int(jcd),
                    "race_no":  rno,
                    "boat_num": b,
                    "racer_name":        race_list_data[b]["racer_name"],
                    "national_win_rate": float(race_list_data[b]["national_win_rate"]),
                    "motor_2ren":        float(race_list_data[b]["motor_2ren"]),
                    "wind_speed":  wind_speed,
                    "wave_height": wave_height,
                    "exhibition_time": float(before_data[b]["exhibition_time"]),
                    "tilt":            float(before_data[b]["tilt"]),
                })
            except (ValueError, TypeError):
                pass

    return pd.DataFrame(rows)


live_df = get_live_data(TARGET_DATE, JCD, TARGET_RACE)

if len(live_df) != 6:
    print("❌ 直前情報がまだ公開されていないか、欠場艇があります。")
    print("展示航走後（レース開始約20分前）に再実行してください。")
    exit()

live_df = add_features(live_df)

# ── 予測と買い目出力 ──────────────────────────────────────────
print(f"=== 📊 AI確率分析 ({TARGET_RACE}R) ===")
probs = model.predict_proba(live_df[FEATURES])
live_df['prob_1st'] = probs[:, 0]
live_df['prob_2nd'] = probs[:, 1]
live_df['prob_3rd'] = probs[:, 2]

for _, row in live_df.iterrows():
    print(
        f"{int(row['boat_num'])}号艇 [{row['racer_name']:　<4}]: "
        f"1着 {row['prob_1st']*100:4.1f}% | "
        f"2着 {row['prob_2nd']*100:4.1f}% | "
        f"3着 {row['prob_3rd']*100:4.1f}%  "
        f"(展示: {row['exhibition_time']})"
    )

print("\n=== 💡 最終ジャッジ ===")
boat1_prob   = live_df[live_df['boat_num'] == 1]['prob_1st'].values[0]
non1         = live_df[live_df['boat_num'] != 1]
boat_1st     = non1.loc[non1['prob_1st'].idxmax()]
racer_1st    = boat_1st['racer_name']
pivot_boat   = int(boat_1st['boat_num'])

if boat1_prob >= 0.50:
    print(f"⚠️ 1号艇の勝率が {boat1_prob*100:.1f}% と高いため【 見 送 り 】を推奨します。")
elif boat_1st['prob_1st'] < THRESHOLD_1ST:
    print(
        f"⚠️ 穴候補の {pivot_boat}号艇 ({racer_1st}) の勝率が "
        f"{boat_1st['prob_1st']*100:.1f}% と基準（{THRESHOLD_1ST*100:.0f}%）未満のため【 見 送 り 】を推奨します。"
    )
else:
    print(
        f"🔥 チャンス！ {pivot_boat}号艇 ({racer_1st}) が1号艇を沈める確率: "
        f"{boat_1st['prob_1st']*100:.1f}%"
    )
    remain = live_df[live_df['boat_num'] != pivot_boat]
    boats_2nd = remain.nlargest(NUM_2ND, 'prob_2nd')['boat_num'].astype(int).tolist()
    boats_3rd = remain.nlargest(NUM_3RD, 'prob_3rd')['boat_num'].astype(int).tolist()

    tickets = [
        f"{pivot_boat}-{b2}-{b3}"
        for b2 in boats_2nd
        for b3 in boats_3rd
        if b2 != b3
    ]

    print(f"\n💰 【穴狙い推奨買い目（3連単 {len(tickets)}点）】")
    for t in tickets:
        print(f" ☑️ {t}")
    print("\n※統計上 回収率239.4% の期待値の設定です。")
