import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import unicodedata
from datetime import datetime, timedelta
import re

START_DATE   = "20230101"
END_DATE     = "20231231"
CSV_FILENAME = f"race_results_{START_DATE}_to_{END_DATE}_ALL.csv"

print(f"=== 🚢 全国24場データ収集 ({START_DATE}〜{END_DATE}) ===")
print(f"保存先: {CSV_FILENAME}")

# course 列を追加（beforeinfo の行順がコース順であることを利用）
header = [
    'date', 'stadium', 'race_no', 'rank', 'boat_num', 'course',
    'racer_id', 'racer_name', 'racer_class', 'national_win_rate',
    'motor_2ren', 'weather', 'wind_speed', 'wave_height',
    'exhibition_time', 'tilt', 'race_time', 'payoff',
]

# ── 初期化 or 再開ポイント検出 ─────────────────────────────────
current_date = datetime.strptime(START_DATE, '%Y%m%d')

if not os.path.exists(CSV_FILENAME):
    pd.DataFrame(columns=header).to_csv(CSV_FILENAME, index=False)
else:
    existing_df = pd.read_csv(CSV_FILENAME)
    if not existing_df.empty:
        last_int = int(existing_df['date'].max())
        last_dt  = datetime.strptime(str(last_int), '%Y%m%d')
        start_dt = datetime.strptime(START_DATE, '%Y%m%d')
        if last_dt >= start_dt:
            existing_df = existing_df[existing_df['date'] < last_int]
            existing_df.to_csv(CSV_FILENAME, index=False)
            current_date = last_dt
            print(f"既存データを検出。{last_int} から再収集します（部分収集対策）。")


def get_race_result(date, jcd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={date}"
    try:
        response = requests.get(url, timeout=10)
        soup     = BeautifulSoup(response.content, 'html.parser')

        if "データがありません" in soup.text or "中止" in soup.text:
            return None

        weather = wind_speed = wave_height = ""
        try:
            for cls, attr, unit in [
                ("is-weather", "weather", ""),
                ("is-wind",    "wind_speed", "m"),
                ("is-wave",    "wave_height", "cm"),
            ]:
                div = soup.find("div", class_=cls)
                if div:
                    s = div.find("span", class_="weather1_bodyUnitLabelData")
                    if s:
                        val = s.get_text(strip=True).replace(unit, '')
                        locals()[attr]  # unused, just to check
                        if cls == "is-weather": weather     = val
                        if cls == "is-wind":    wind_speed  = val
                        if cls == "is-wave":    wave_height = val
        except Exception:
            pass

        # weather/wind/wave を一括取得（上のロジックを修正）
        weather = wind_speed = wave_height = ""
        try:
            d = soup.find("div", class_="is-weather")
            if d:
                s = d.find("span", class_="weather1_bodyUnitLabelData")
                if s: weather = s.get_text(strip=True)
            d = soup.find("div", class_="is-wind")
            if d:
                s = d.find("span", class_="weather1_bodyUnitLabelData")
                if s: wind_speed = s.get_text(strip=True).replace('m', '')
            d = soup.find("div", class_="is-wave")
            if d:
                s = d.find("span", class_="weather1_bodyUnitLabelData")
                if s: wave_height = s.get_text(strip=True).replace('cm', '')
        except Exception:
            pass

        payoff = None
        try:
            for block in soup.find_all(["tr", "tbody"]):
                is_3rentan = "3連単" in block.get_text() or "３連単" in block.get_text()
                if not is_3rentan:
                    for img in block.find_all("img"):
                        if "3連単" in img.get("alt", "") or "３連単" in img.get("alt", ""):
                            is_3rentan = True
                            break
                if is_3rentan:
                    for td in block.find_all(["td", "th"]):
                        td_text = td.get_text(strip=True)
                        if "円" in td_text or "¥" in td_text:
                            m = re.search(r'[¥]?([0-9,]+)円?', td_text)
                            if m:
                                payoff = int(m.group(1).replace(',', ''))
                                break
                    if payoff:
                        break
        except Exception:
            pass

        table = soup.find("table", class_="is-w495")
        if not table:
            return None

        results = []
        for row in table.find_all("tbody"):
            cols = row.find_all("td")
            if len(cols) >= 4:
                try:
                    rank_str = unicodedata.normalize('NFKC', cols[0].get_text(strip=True))
                    if not rank_str.isdigit():
                        continue
                    rank     = int(rank_str)
                    boat_num = int(unicodedata.normalize('NFKC', cols[1].get_text(strip=True)))
                    parts    = cols[2].get_text(separator=" ", strip=True).split()
                    racer_id   = parts[0] if parts else ""
                    racer_name = "".join(parts[1:]).replace('　', '') if len(parts) > 1 else ""
                    results.append({
                        "date": date, "stadium": int(jcd), "race_no": rno,
                        "rank": rank, "boat_num": boat_num,
                        "racer_id": racer_id, "racer_name": racer_name,
                        "weather":     weather,
                        "wind_speed":  float(wind_speed)  if wind_speed  else None,
                        "wave_height": float(wave_height) if wave_height else None,
                        "race_time":   cols[3].get_text(strip=True),
                        "payoff":      payoff if rank == 1 else None,
                    })
                except Exception:
                    pass
        return results if results else None
    except requests.exceptions.RequestException:
        return None


def get_before_info(date, jcd, rno):
    """
    beforeinfo テーブルの行はコース順（1〜6）で並ぶため、
    行インデックスをコース番号として記録する。
    cols[0] = 艇番（ボートレース公式のbeforeinfo構造より）
    """
    url  = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={date}"
    data = {}
    try:
        soup         = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        course_order = 0
        for tbody in soup.find_all("tbody"):
            for row in tbody.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 6:
                    try:
                        raw = unicodedata.normalize('NFKC', cols[0].get_text(strip=True))
                        if not raw or not raw[0].isdigit():
                            continue
                        boat_num     = int(raw[0])
                        course_order += 1
                        ex_str = cols[4].get_text(strip=True)
                        ti_str = cols[5].get_text(strip=True)
                        data[boat_num] = {
                            "course":         course_order,
                            "exhibition_time": float(ex_str) if ex_str else None,
                            "tilt":            float(ti_str) if ti_str else None,
                        }
                    except Exception:
                        pass
    except requests.exceptions.RequestException:
        pass
    return data


def get_race_list(date, jcd, rno):
    url  = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={date}"
    data = {}
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        for tbody in soup.find_all("tbody", class_="is-fs12"):
            cols = tbody.find_all("td")
            if len(cols) >= 8:
                try:
                    raw = unicodedata.normalize('NFKC', cols[0].get_text(strip=True))
                    if not raw or not raw[0].isdigit():
                        continue
                    boat_num    = int(raw[0])
                    class_parts = cols[2].get_text(separator=" ", strip=True).split()
                    racer_class = class_parts[-1] if class_parts else ""
                    rate_parts  = cols[5].get_text(separator=" ", strip=True).split()
                    nat_rate    = rate_parts[0] if rate_parts else None
                    motor_parts = cols[7].get_text(separator=" ", strip=True).split()
                    motor_2ren  = motor_parts[1] if len(motor_parts) > 1 else None
                    data[boat_num] = {
                        "racer_class":       racer_class,
                        "national_win_rate": float(nat_rate)   if nat_rate   else None,
                        "motor_2ren":        float(motor_2ren) if motor_2ren else None,
                    }
                except Exception:
                    pass
    except requests.exceptions.RequestException:
        pass
    return data


end_date_obj = datetime.strptime(END_DATE, '%Y%m%d')

while current_date <= end_date_obj:
    target     = current_date.strftime('%Y%m%d')
    daily_data = []
    print(f"\n📅 【{target}】のデータを取得中...")

    for jcd_num in range(1, 25):
        jcd = f"{jcd_num:02d}"
        try:
            res = requests.get(
                f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno=1&jcd={jcd}&hd={target}",
                timeout=10,
            )
            time.sleep(1)
            if "データがありません" in BeautifulSoup(res.content, 'html.parser').text:
                continue
        except Exception:
            time.sleep(1)
            continue

        print(f" 🚤 会場: {jcd} ...", end="", flush=True)

        for rno in range(1, 13):
            results = get_race_result(target, jcd, rno)
            time.sleep(1)
            if not results:
                continue

            before    = get_before_info(target, jcd, rno)
            time.sleep(1)
            race_list = get_race_list(target, jcd, rno)
            time.sleep(1)

            for row in results:
                b = row["boat_num"]
                row["course"]           = before.get(b, {}).get("course")
                row["exhibition_time"]  = before.get(b, {}).get("exhibition_time")
                row["tilt"]             = before.get(b, {}).get("tilt")
                row["racer_class"]      = race_list.get(b, {}).get("racer_class", "")
                row["national_win_rate"] = race_list.get(b, {}).get("national_win_rate")
                row["motor_2ren"]        = race_list.get(b, {}).get("motor_2ren")
                daily_data.append(row)

        print(" 完了！")

    if daily_data:
        pd.DataFrame(daily_data).reindex(columns=header).to_csv(
            CSV_FILENAME, mode='a', header=False, index=False,
        )
        print(f" 💾 {target} を保存しました。")

    current_date += timedelta(days=1)

print("\n🎉 全国24場のデータ収集が完了しました！")
