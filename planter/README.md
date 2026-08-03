# planter — おしゃれな植木鉢の 3D モデルジェネレータ

パラメトリックに植木鉢（と受け皿）の 3D モデルを生成し、**STL / OBJ** で書き出します。
**純 Python・標準ライブラリのみ**で動くので、numpy も CAD も不要です。

```bash
cd planter
python3 planter.py --preview preview.svg
```

```
[planter] 高さ 132.0mm / 最大径 129.2mm / 底径 72.0mm
          三角形 93,696 / 材料 163.8cm3 / 容量 950mL / 最薄肉厚 2.21mm
          メッシュ検証: OK (watertight)
          -> ./planter.stl (4575 KB)
[saucer]  高さ 22.4mm / 最大径 102.2mm / 底径 82.1mm
          ...
          -> ./planter_saucer.stl (4538 KB)
[preview] -> preview.svg (2780 KB)
```

生成されるもの:

- `planter.stl` — 本体（ねじれた縦溝・くびれた足元・中央に排水穴）
- `planter_saucer.stl` — 鉢がぴったり収まる受け皿（穴なし）
- `preview.svg` — 外部ビューアなしで形を確認するためのプレビュー画像

## スタイル

`--style` でデザインを選べます。

| スタイル | 見た目 |
| --- | --- |
| `fluted` (既定) | ねじれた縦溝。口元は溝が消えてすっきりした襟になる |
| `ribbed` | 細かい縦リブが並ぶコンクリート鉢風 |
| `faceted` | 9 角形のローポリ。`--segments` で角数を変更 |
| `wave` | 横方向の波（`--wave-count` / `--wave-depth`） |
| `squircle` | 角丸四角の断面（`--squircle` で丸み調整） |
| `smooth` | 装飾なしの回転体 |

```bash
python3 planter.py --style faceted --segments 7          # 七角形の鉢
python3 planter.py --style ribbed --height 90 --top-diameter 100
python3 planter.py --style fluted --twist 60 --flutes 18 --flute-depth 3
python3 planter.py --style squircle --squircle 6 --preview sq.svg
```

## 主なオプション

| オプション | 説明 | 既定 |
| --- | --- | --- |
| `--height` | 高さ mm | 132 |
| `--top-diameter` / `--bottom-diameter` | 上端 / 底の直径 mm | 124 / 78 |
| `--wall` | 最小肉厚 mm | 2.2 |
| `--floor` | 底の厚み mm | 5.0 |
| `--drain-diameter` | 排水穴の直径 mm（0 で穴なし） | 14 |
| `--lip` | 口元の張り出し mm | 2.6 |
| `--belly` | 胴の膨らみ（上端半径比） | 0.045 |
| `--foot-inset` | 底の絞り込み mm | 3.0 |
| `--flutes` / `--flute-depth` / `--flute-sharpness` | 縦溝の本数 / 深さ / 鋭さ | 26 / 2.4 / 1.8 |
| `--twist` | 上端までのねじれ角（度） | 26 |
| `--scale` | 寸法だけを一括倍率変更（分割数はそのまま） | 1.0 |
| `--segments` / `--rows` | 円周 / 高さ方向の分割数 | 192 / 120 |
| `--fluted-inside` | 内側にも溝をつけて材料を節約 | off |
| `--no-saucer` | 受け皿を作らない | off |
| `--format` | `stl` / `stl-ascii` / `obj` / `all` | `stl` |
| `--preview FILE.svg` | SVG プレビューを書き出す | — |

`python3 planter.py --help` で全オプションが出ます。

## 出力の検証

書き出す前に必ずメッシュを検査し、問題があれば終了コード 3 で中断します。

- **水密性** — 全ての有向辺がちょうど 1 回ずつ現れ、逆向きの相方を持つこと（多様体）
- **法線の向き** — 符号付き体積が正であること（＝外向き）
- **肉厚** — 溝の底も含めた最薄部が `--wall` を下回らないこと
- **成立性** — 内側空間が潰れていないか、排水穴が底からはみ出していないか

材料体積（cm³）と土の容量（mL）も表示されるので、フィラメント量や鉢のサイズ感の目安になります。

## 3D プリントの目安

- 底を下にしてそのまま置くだけ。**サポート不要**（最急の張り出しは垂直から約 28°、
  受け皿でも約 41° で、一般的な限界の 45° に収まります）
- 層厚 0.2mm / 壁 3 周 / インフィル 15% あたりが無難。既定サイズで PLA 約 200g
  （中実体積 164cm³ + 受け皿 35cm³）
- 水漏れが気になる場合は `--wall 2.8` 程度に上げるか、内側に防水塗装を
- 受け皿は鉢の足回り + `--saucer-clearance`（既定 3mm）の余裕を自動で確保します

## ライブラリとして使う

```python
from dataclasses import replace
import planter as P

params = replace(P.Params(), height=180, top_radius=70, twist=45, flute_count=20)
mesh, info = P.build_planter(params)
print(info["capacity_ml"], mesh.check())
P.write_stl(mesh, "tall.stl")
P.render_svg([(mesh, "#8d9b8a")], "tall.svg", azimuth=40, elevation=18)
```

## テスト

```bash
python3 test_planter.py
```

ジオメトリ（全スタイルの水密性・肉厚・排水穴・受け皿の嵌合）、エクスポータ、CLI を
16 ケースで検証します。

## 仕組み

形は 2 つの半径関数だけで決まります。

- `outer_radius(θ, t)` = 輪郭 `base_profile(t)` × 断面 `section_mod(θ)` − 溝 `groove(θ, t)`
- `inner_radius(θ, t)` = 輪郭 − 必要肉厚 × 法線補正

法線補正は、半径方向に 1mm 動いても実際の肉厚は 1mm 増えない（テーパや波で面が傾いて
いる）ぶんを `|∇(r − R)|` で補正するものです。これで口元が広がっていても溝が深くても、
肉厚は指定値を下回りません。溝は上下端でフェードアウトさせているので、縁と底は常に
きれいな平面リングになります。

メッシュは輪（リング）を順に縫い合わせるだけで作ります。三角形の向きは「リングを渡す
順番」だけで決まるので、外向き法線と水密性が構造的に保証されます。

```
リング接合の順番            法線の向き
外壁   下 → 上              外向き
縁     外 → 内              上向き
内壁   上 → 下              軸向き（内向き）
床上面 内壁 → 排水穴        上向き
排水穴 上 → 下              軸向き
底面   排水穴 → 外周        下向き
```

プレビューはこのメッシュを直交投影し、裏面を除去して奥から順に SVG のポリゴンとして
描いたものです（フラットシェーディング＋クリース角 26° の頂点法線スムージング）。
ブラウザでそのまま開けます。
