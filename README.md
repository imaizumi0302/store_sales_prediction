# Store Sales Prediction

## プロジェクト概要

Kaggleのコンペを題材にデータ分析を行いました。
エクアドルの食料品小売業者の各店舗の各商品ごとの販売個数を予測するプロジェクトです。
具体的には、過去数年間の各店舗、各商品ごとの販売個数データから、ある一定期間の各店舗、買う商品ごとの販売個数を予測し、
商品の在庫管理などに生かすデータ分析です。

---

## 背景

取り組みやすい内容であると同時に、複数のCSVデータを組み合わせて分析をする必要のあるプロジェクトであったため、
このプロジェクトに取り組みました。：

* データの前処理
* 時系列データの特徴量エンジニアリング
* 時系列データにおける交差検証法によるモデル評価
* ハイパーパラメータチューニング（Optuna）
* SHAP値の可視化

---

## 技術スタック

* 言語: Python 3.12
* ライブラリ: pandas, numpy, scikit-learn, lightgbm, imbalanced-learn, Streamlit, Optuna
* データベース: SQLite
* バージョン管理: Git, GitHub

---

## フォルダ・ファイル構成

```
soccer_pipeline/
├─ data/                     # データファイル格納
│  ├─ premier_league.csv
│  ├─ processed_data.csv
│  └─ latest_predictions.json
├─ db/                       # SQLite データベース
│  └─ matches.db
├─ logs/                     # 実行ログ
│  ├─ pipeline.log
│  └─ service_main.log
├─ models/                   # 学習済みモデル
│  ├─ final_model.pkl
│  └─ model_lgb_fold*.pickle
├─ notebooks/                # Jupyter Notebook
│  ├─ EDA_notebook.ipynb
│  ├─ processed_notebook.ipynb
│  └─ season_data_process.ipynb
├─ src/                      # スクリプト
│  ├─ app.py
│  ├─ data_fetcher2.py
│  └─ prediction_pipeline1.py
└─ .gitignore
```

---

## 主要スクリプトの概要

| スクリプト                         | 役割                                  | 出力                        |
| ----------------------------- | ----------------------------------- | ------------------------- |
| `src/data_fetcher2.py`        | API-FOOTBALL から試合データを取得し、SQLite に保存 | `matches.db`              |
| `src/prediction_pipeline1.py` | データ結合・前処理・特徴量作成・学習・予測               | `latest_predictions.json` |
| `src/app.py`                  | Streamlit でダッシュボード表示                | ブラウザ上の可視化 UI              |

---

## 実行手順

### 1. 必要なパッケージのインストール

```bash
pip install -r "requirements.txt"
```

### 2. データベース準備

API から試合データを取得して SQLite に保存：

```bash
python src/data_fetcher2.py
```

※実行時に自身の API キーを設定してください。

### 3. メインパイプライン実行

```bash
python src/prediction_pipeline1.py
```

### 4. Streamlit アプリ起動

```bash
streamlit run src/app.py
```

---

## 出力結果

| 出力                               | 内容                                   |
| -------------------------------- | ------------------------------------ |
| `data/latest_predictions.json`   | Streamlit 用の予測結果 JSON                |
| `db/matches.db/matches`          | 各試合の実際の試合結果                          |
| `db/matches.db/match_statistics` | 実施済みの試合の統計データ                        |
| `db/matches.db/predictions`      | 予測結果 (H:ホーム勝利, D:引き分け, A:アウェイ勝利) と確率 |
| `models/final_model.pkl`         | 作成された学習済みモデル                         |
| `Streamlit UI`                   | 試合予測結果、発生確率、確信度、モデル精度をブラウザ上で確認可能     |

---

## モデル概要

* **使用モデル:** LightGBM
* **目的:** プレミアリーグ試合結果の 3 クラス分類（H:ホーム勝利, D:引き分け, A:アウェイ勝利）
* **評価指標:** Accuracy, F1-score (macro & weighted), log_loss

  * F1-score: 不均衡データに対する予測の正確さを評価
  * log_loss: 予測確率の誤差を評価
* **特徴量:** 過去の試合の得点/失点、チーム勝率、ホーム/アウェイ情報、シーズン勝ち点、昨シーズン情報(順位、得点、失点 等)
* **学習方法:** KFold 3-fold クロスバリデーション
* **不均衡データ対策:** `class_weight='balanced'`

---

## 今後の改善点

* モデル精度向上（特徴量追加、ハイパーパラメータ調整）
* Streamlit ダッシュボードの UI 改善
* リアルタイムデータ取得への対応

---

## 注意点

* `DB_PATH` や `SEASON_DATA_PATH` はスクリプト内で絶対パスを設定してください
* `data_fetcher2.py` 実行時に自身の API キーを設定してください
* GitHub にアップロードする際は `.gitignore` で不要ファイルを除外しています

---

## ライセンス

このプロジェクトは **MIT ライセンス** の下で公開されています。
詳細は [LICENSE](./LICENSE) ファイルをご覧ください。
