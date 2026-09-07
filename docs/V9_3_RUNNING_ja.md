# v9.3をローカルで実行する

実行するファイルは `notebooks/tsAIME_APR_all_experiments_v9_3.ipynb` である。
v9.2のノートブックと結果は残している。
旧版のコード一式は `archives/v9_2_source_before_20260906.zip` にも保存した。
旧ノートブックはtsaime 0.3.1を要求するため、旧版を再実行するときはアーカイブを別のフォルダに展開し、別の環境を使う。

## インストールと実行

`codev9` で次を実行する。

```bash
python -m pip install -e '.[all]'
python -m jupyter lab
```

新しいカーネルでv9.3のノートブックを開き、先頭からすべて実行する。
ソースのフォルダをノートブックが見つければ、GitHubへ接続してtsaimeを取得する必要はない。
科学計算ライブラリが不足している場合はインストールが必要になる。
初回のデータ取得にはインターネット接続が必要である。
元データのキャッシュが存在すれば再利用する。

既定は `publication` で、北京12地点とつくばを実行する。
実行モードのセルで `CONFIG.run_mode` が `publication` であることを確認する。
過去に `TSAIME_V9_RUN_MODE=smoke` を設定した環境では、その設定が優先される。
短い動作確認を行う場合は、実行モードのセルを次のように変更する。

```python
CONFIG = APRExperimentConfig(run_mode="smoke").resolved()
```

smokeは北京2地点、つくば、少ない再標本化回数による動作確認であり、論文の結果として使わない。
publicationに戻したら、実行セルから最後まで改めて実行する。

## 保存場所

毎回、次の場所に新しいフォルダを作る。

```text
results/v9_3_apr/<実行時刻>/
  outputs/
    figures/                 PDFとPNG
    csv/                     完全な結果表とTABLE_INDEX.csv
    tex/                     明示的な要約表、booktabsとlongtableを使用
    logs/                    実行条件、ソースのハッシュ、失敗記録
    README_RESULTS.md
    tsAIME_APR_v9_3_outputs_<実行時刻>.zip
  private_traces/             時刻ごとの予測と再構成、較正境界、標準化
```

元データは既存の `results/v9_apr/data/` を再利用する。
別のキャッシュを使う場合は `TSAIME_DATA_CACHE` 環境変数を指定する。
過去の出力を消して書き直す処理は行わない。
実行が途中で止まっても、終了した地点のCSVと再現用トレースは残る。

## 結果の確認

`TABLE_INDEX.csv` で表の名前と分析内容を確認する。
`computational_checks` の表は、地点数、数理的一致、時間境界、処理失敗を検査する。
有意差、他手法への勝利、小さい回復誤差はPASSの条件にしない。
`run_mode=publication` と記録された結果だけを論文用の候補とする。
さらに、結果が主張を支持するかを別に判断する。

気象変数の主比較は `inverse_rolling` の `validation_selected_single`、`marginal_mean`、`past_mean` である。
`marginal_sum_ablation` は重複した出力情報を補正しない対照であり、相関を使う方法全般の代表ではない。
`primary_interval=True` が主表示の28日ブロックで、7日と14日は感度分析である。
各区間は学習済みモデルを条件とする個別の95%区間であり、多重比較を調整した同時区間ではない。

## 公開前の区別

研究コードはPolyForm Noncommercial 1.0.0を維持する。
元データの利用条件はこのソフトウェアライセンスとは別である。
NIESのファイルと公式ページのライセンス表記の相違は、コード変更では解決しないため公開前に確認する。
観測値を含む `private_traces/` はZIPに入れず、GitHub等へも条件確認なしで公開しない。
GitHubへの更新やコミットは自動では行わない。
