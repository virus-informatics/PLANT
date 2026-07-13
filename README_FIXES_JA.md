# PLANT 修正版

今回指定された修正を反映したファイル一式です。

## 反映した修正

1. `--max-length` はraw amino-acid length（既定値329）のままとし、tokenizationには `token_max_length = max_length + 2` を使用。
2. season split時、virus-only sequence poolをmetadataの`date`でcutoff以前に限定してからdownsampling。
   - `1996--`、`1996-`、`1996`などは1996年全体のintervalとして扱います。
   - 最新の可能日がcutoffを越えるpartial date、欠損・解釈不能dateは保守的に除外します。
   - 除外行は `virus_only_sequence_pool_excluded_by_cutoff.csv` に保存します。
3. `use_systematic_error=False` のpairwise inferenceでは、model forward内でsystematic-error経路を明示的に無効化。
4. `freeze_esm=True` のとき、frozen ESM backboneを常にeval modeに固定。
5. `category_mappings.json` を全データではなくtraining splitのみから作成。
6. random splitを概ね train/validation/test = 80/10/10 に変更。
7. OneHotEncoderをmodule-global変数から各`semanticESM` instanceの属性へ移動。
8. 前回確認したsingleton virus-only batchについて、differentiable zeroを返すようにして`loss.backward()`停止を防止。
9. CSV読込でbare yearが`1996.0`のようなfloatになった場合も、年全体のintervalとして解釈。
10. systematic-error encoderが欠損・部分欠損している場合、silentな無補正や行列shape errorではなく明示的な例外を送出。
11. `freeze_esm=True`では構築直後からESMをeval modeにし、`semanticESM.from_pretrained()`もHugging Face同様にeval modeで返却。
12. 最終バッチが1件だけの場合、そのsingleton batchのみを除外し、勾配ゼロのstepがoptimizer/schedulerを消費しないよう修正。通常の不完全バッチ（2件以上）は維持。

## 主なファイル

- `Training/PLANT_train_with_module_time_split_hparam_k10_virus_only_stratified_season_or_full.py`
- `src/plant/model.py`
- `src/plant/inference.py`
- `src/plant/__init__.py`

`src/plant/data.py`と`src/plant/training.py`は、配置しやすいよう元ファイルを同梱しています。

## 確認済み項目

- 全Pythonファイルの構文コンパイル
- `1996--`を含むpartial-date cutoff filter
- 80/10/10 split
- `apply_systematic_error=False`時の2つのdistance出力の一致
- frozen ESMのeval mode維持
- singleton virus-only batchのbackward
- 異なるmodel instance間でencoder stateが共有されないこと
- `1996.0`として読まれたbare yearのcutoff処理
- encoder欠損・部分欠損時の明示的エラー
- save/load後にeval modeが維持されること
- 最終singleton batchのみが除外され、通常のpartial batchは保持されること

実際の3B ESMを用いたend-to-end学習は、この実行環境に`transformers`がないため未実施です。
