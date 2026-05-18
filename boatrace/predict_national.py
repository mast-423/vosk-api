"""
predict_national.py

旧ロジックとの主な変更点:
  1. 独立分類 → Plackett-Luce でレース内確率を正規化
  2. 閾値フィルター → 3連単オッズを取得して期待値(EV)で買い目を選別
  3. Kelly基準で推奨ベット比率を計算
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import lightgbm as lgb
import joblib
import unicodedata
import os
import re
import warnings
from datetime import datetime

from utils import add_features, FEATURES, plackett_luce_prob, kelly_fraction

warnings.filterwarnings('ignore')

TARGET_DATE = datetime.today().strftime('%Y%m%d')
print(f"=== 🤖 全国対応版・EV最大化AI予想システム ({TARGET_DATE}) ===")

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

print(f"\n🚀 【会場: {JCD} / {TARGET_RACE}R】 予想開始...\n")

# 得意会場フィルター: simulate_national_advanced.py 実行後に設定
GOOD_STADIUMS: list = []
MIN_EV = 0.0        # 期待値の最低ライン（0 = ゼロ以上すべて表示）
KELLY_CAP = 0.05    # 1点あたりの最大ベット比率（5%）
HALF_KELLY = True   # 実戦では半Kelly推奨（モデル不確実性を考慮）

if GOOD_STADIUMS and JCD not in GOOD_STADIUMS:
    print(f"⚠️ 会場 {JCD} はAIの得意会場リスト外です。")
    if input("続けますか？ (y/n): ").strip().lower() != 'y':
        exit()

# ── モデルの準備 ───────────────────────────────────────────────
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
        n_estimators=100, learning_rate=0.05, random_state=42,
        objective='multiclass',
        class_weight='balanced',
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
                name  = a_tag.get_text(strip=True).replace('　', '') if a_tag else "不明"
                rate_parts  = cols[5].get_text(separator=" ", strip=True).split()
                motor_parts = cols[7].get_text(separator=" ", strip=True).split()
                race_list_data[b] = {
                    "racer_name":        name,
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
        w = soup_before.find("div", class_="is-wind")
        if w:
            s = w.find("span", class_="weather1_bodyUnitLabelData")
            if s: wind_speed = float(s.get_text(strip=True).replace('m', ''))
        v = soup_before.find("div", class_="is-wave")
        if v:
            s = v.find("span", class_="weather1_bodyUnitLabelData")
            if s: wave_height = float(s.get_text(strip=True).replace('cm', ''))
    except Exception:
        pass

    before_data  = {}
    course_order = 0
    for tbody in soup_before.find_all("tbody"):
        for row in tbody.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 6:
                try:
                    raw = unicodedata.normalize('NFKC', cols[0].get_text(strip=True))
                    if not raw or not raw[0].isdigit():
                        continue
                    b = int(raw[0])
                    if b not in range(1, 7):
                        continue
                    course_order += 1
                    before_data[b] = {
                        "course":          course_order,
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


def get_odds(date, jcd, rno) -> dict:
    """
    3連単オッズを {(1着艇, 2着艇, 3着艇): 払戻金額} で返す。
    例: {(2,1,3): 4200, ...}
    払戻は100円賭けたときの払戻金（テラ銭控除後）。
    """
    url  = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd}&hd={date}"
    odds = {}
    sep  = re.compile(r'[-−ー]')
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            for i, cell in enumerate(cells):
                raw = unicodedata.normalize('NFKC', cell.get_text(strip=True))
                parts = sep.split(raw)
                if (len(parts) == 3
                        and all(p.isdigit() and 1 <= int(p) <= 6 for p in parts)):
                    combo = (int(parts[0]), int(parts[1]), int(parts[2]))
                    for j in range(i + 1, min(i + 3, len(cells))):
                        pay_raw = unicodedata.normalize('NFKC', cells[j].get_text(strip=True))
                        pay_num = re.sub(r'[^\d]', '', pay_raw)
                        if pay_num and int(pay_num) >= 100:
                            odds[combo] = int(pay_num)
                            break
    except requests.exceptions.RequestException:
        pass
    return odds


# ── 直前情報取得 ───────────────────────────────────────────────
live_df = get_live_data(TARGET_DATE, JCD, TARGET_RACE)

if len(live_df) != 6:
    print("❌ 直前情報が公開されていないか、欠場艇があります。")
    print("展示航走後（発走約20分前）に再実行してください。")
    exit()

live_df = add_features(live_df)

# ── モデル推論 ─────────────────────────────────────────────────
probs = model.predict_proba(live_df[FEATURES])
live_df = live_df.copy()
live_df['prob_1st'] = probs[:, 0]
live_df['prob_2nd'] = probs[:, 1]
live_df['prob_3rd'] = probs[:, 2]

# Plackett-Luce に使う "強度" スコア (softmax 前の生スコアを使う)
pl_scores = dict(zip(live_df['boat_num'].astype(int), live_df['prob_1st']))
# レース内 softmax で「見やすい確率」を計算（表示用）
total_s = sum(pl_scores.values())
norm_1st = {b: s / total_s for b, s in pl_scores.items()} if total_s else pl_scores

print(f"=== 📊 AI確率分析 ({TARGET_RACE}R) ===")
for _, row in live_df.iterrows():
    b   = int(row['boat_num'])
    p1  = norm_1st[b] * 100
    p2  = row['prob_2nd'] * 100
    p3  = row['prob_3rd'] * 100
    print(
        f"{b}号艇 [{row['racer_name']:　<4}]: "
        f"1着 {p1:4.1f}% | 2着 {p2:4.1f}% | 3着 {p3:4.1f}%  "
        f"(展示: {row['exhibition_time']})"
    )

# ── オッズ取得 ─────────────────────────────────────────────────
print("\nオッズを取得中...")
odds_dict = get_odds(TARGET_DATE, JCD, TARGET_RACE)

print(f"\n=== 💡 最終ジャッジ ===")

if not odds_dict:
    # ── フォールバック: 旧ロジック（オッズ未取得時） ─────────────
    print("⚠️ オッズ未取得。発走10分前に再実行してください。")
    print("（参考）最も1着確率が高い艇:")
    top = max(norm_1st.items(), key=lambda x: x[1])
    print(f"  → {top[0]}号艇 {top[1]*100:.1f}%")
else:
    # ── メインロジック: EV × Kelly ─────────────────────────────
    ev_list = []
    for combo, payout in odds_dict.items():
        prob = plackett_luce_prob(pl_scores, combo)
        # 期待値 = prob × 払戻 / 100 - 1
        ev = prob * (payout / 100.0) - 1.0
        if ev > MIN_EV:
            kf_raw  = kelly_fraction(prob, payout)
            kf_used = min(kf_raw * (0.5 if HALF_KELLY else 1.0), KELLY_CAP)
            ev_list.append({
                'ticket':  f"{combo[0]}-{combo[1]}-{combo[2]}",
                'payout':  payout,
                'prob':    prob,
                'ev':      ev,
                'kelly':   kf_used,
            })

    ev_list.sort(key=lambda x: x['ev'], reverse=True)

    if not ev_list:
        print("⚠️ 期待値プラスの買い目が見つかりません。【 見 送 り 】を推奨します。")
    else:
        print(f"🔥 期待値プラスの買い目: {len(ev_list)}点")
        print()
        print(f"{'買い目':<10} {'払戻':>7} {'AI確率':>7} {'期待値':>8} {'Kelly推奨':>10}")
        print("-" * 50)
        for r in ev_list[:15]:
            kpct = r['kelly'] * 100
            print(
                f"☑️ {r['ticket']:<8} "
                f"{r['payout']:>7,}円 "
                f"{r['prob']*100:>6.2f}% "
                f"{r['ev']*100:>+7.1f}% "
                f"{kpct:>9.2f}%"
            )
        print()
        print("※ Kelly推奨は資金の何%を賭けるべきかの目安（半Kelly適用済）")
        print("※ 期待値 = AI予測確率 × 払戻 / 100 - 1")
        if len(ev_list) > 15:
            print(f"（他 {len(ev_list)-15}点 省略）")
