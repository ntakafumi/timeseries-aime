# v9.4の実行と公開準備

## 使うファイル

`notebooks/tsAIME_APR_all_experiments_v9_4.ipynb` を使用する。
データ監査、合成実験、予測、説明の追加検証、図表、ZIP生成を含む。
APR用の別実験モジュールには依存しない。
共通パッケージは `src/tsaime/` の0.3.3である。
0.3.2からの変更は暦グリッド判定の日時精度互換性のみで、逆作用素の数式は同じである。

## ローカルJupyterでの実行

ターミナルの現在位置をcodev9にして、次を実行する。

```bash
python -m pip install -e '.[all]'
python -m jupyter lab
```

ノートブックを開き、カーネルを再起動して全セルを順番に実行する。
GitHubへの接続やGitHubからのインストールは不要である。
同じカーネルに旧tsaimeが読み込まれている場合は、再起動が必要である。

検証環境をそろえる場合はPython 3.13の独立環境で次を実行する。

```bash
python -m pip install -r requirements-v9.4.txt
python -m pip install -e . --no-deps
python -m jupyter lab
```

requirements-v9.4.txtは今回の科学計算環境を固定するファイルであり、すべてのOSとPython版での動作を保証するものではない。

## 実行モード

通常はノートブックの設定を変更しない。

| 設定 | 既定値 | 意味 |
|---|---|---|
| RUN_MODE | publication | 論文用の全地点と全設定 |
| EXECUTION_MODE | auto | 適合するv9.3実行を検証して再利用。なければ全実験 |
| SOURCE_RUN | None | codev9内のv9.3保存実行から適合候補を探す |

`reuse` では保存済み実行が必須となり、`full` ではデータ監査から全実験を行う。
明示指定するSOURCE_RUNはZIPではなく、outputsとprivate_tracesのある実行フォルダである。
設定が一致する候補のファイルが壊れていれば停止する。
その場合はログを確認し、意図してfullを選ぶか、正しいSOURCE_RUNを指定する。

初めて公開リポジトリを取得した共同研究者には、非公開の保存済み結果はないため、autoはfullとして実行される。
fullは公開データをダウンロードするためネット接続が必要になる。
ダウンロード元の障害や変更による取得失敗は、成功として扱わず記録する。

短い動作確認の場合だけRUN_MODEをsmokeにする。
smokeの数値を論文結果へ混ぜない。

## 出力

```text
results/v9_4_apr/<run-id>/
├── private_traces/       時刻ごとの観測、予測、再構成。公開対象外
└── outputs/
    ├── figures/          PDFとPNG
    ├── csv/              完全な数値表、目次、ハッシュ
    ├── tex/              明示的に集約したLaTeX表
    ├── logs/             設定、環境、再利用元、エラー
    ├── README_RESULTS.md
    ├── FIGURE_GUIDE.md
    └── tsAIME_APR_v9_4_outputs_<run-id>.zip
```

旧ノートブックと旧結果を上書きしない。
ZIPにはprivate_tracesと生のデータダウンロードを含めない。
事例図には観測値が描画されるため、データ利用条件の確認は図にも必要である。

最初にcomputational_checksとfailuresを確認する。
続いてexplanation_controls、coefficient_stability、reconstruction_stability、episode_summary、claim_evidenceを確認する。
表番号はTABLE_INDEX.csvで確認する。
係数が不安定な変数を、安定した物理要因として解釈しない。
PASSは採択や統計的な優位性を意味しない。

生成したTeX表には `booktabs`、`longtable`、`array` が必要である。
長い検証表は補足資料用であり、46表すべてを本文へ入れる想定ではない。
本文では主要比較を選び、最終的な誌面幅で図表を配置する。

## GitHubへ出すファイル

本体はsrc/tsaime、ノートブックはnotebooks、説明はREADMEとdocs、検証はtestsにある。
v9.2とv9.3は履歴であり、新規実行にはv9.4を使う。
配布用ZIPでは旧ノートブックも出力を除いたコピーにするが、ローカルの実行済み原本は変更しない。

次は公開しない。

- results、private_traces、生データのキャッシュ。
- ローカル監査の作業ファイルや開発途中の出力。
- 個人のパスや実行出力が残ったノートブック。

ソフトウェアはPolyForm Noncommercial License 1.0.0のままである。
データのDOI、version、取得時刻と検証時刻はソフトウェアのライセンスとは別に管理する。
NIESの利用条件の表記不一致は、データ提供者の条件を確認してから解消する。
ソフトウェアDOIは未取得の値を記入しない。
GitHubとZenodoへ実際に公開した後に、確定したタグとDOIを記入する。

## パッケージと公開用ZIP

```bash
python -m pytest
python -m build
python tools/build_release.py
```

パッケージのwheelとtar.gzはdistにできる。
build_release.pyは明示したコードと文書だけを収集し、全ノートブックの実行出力を除いた公開用コピーをreleaseに作る。
元のノートブック、結果、GitHubは変更しない。
追加実験に必要な処理はノートブック自身にあり、この配布ツールは実験の依存先ではない。

## v9.4で固定する範囲

今回の手法は線形ts-AIMEで完結させる。
二次逆回帰は補足的な再構成診断としてのみ残す。
新しいKernel-AIME、データ地域、予測器、探索的なパラメータ調整は追加しない。
結果が仮説を支持しない場合は、次版を作って有利な設定を探すのではなく、原稿の主張を結果に合わせる。
