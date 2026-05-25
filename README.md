主要ファイル

experiment.py — 検証 (3.1) とベンチ (3.2) の実行入口
koyuti.py — 固有値・固有ベクトル（べき乗法 + NumPy）関連
gyoretushiki.py — 行列式計算
renritu.py — 連立方程式解法（ガウス消去）
使い方（ワークスペースのルートで実行）

3.1 の検証だけ:
python experiment.py --mode validate
ベンチマークを n=500 まで実行（要注意: 重い）:
python experiment.py --mode benchmark --max-size 500
