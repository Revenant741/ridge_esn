# ga_esn
拘束条件付きESNの実装
遺伝的アルゴリズムを用いた最適なリザバー層の探索

# プログラムの種類
model.py ESNのみの実行
binde_only_model.py 拘束条件付きESNの実装
binde_bias_model.py バイアス追加とHeの初期化が用いられた拘束条件付きESNの実装
ga.py 遺伝的アルゴリズムを含めたプログラムの実行

# 実行方法

  `$python3 setup.py`
  
  `$docker start -i my-esn`
  
  `$python3 src/train.py`

遺伝的アルゴリズムは
500世代，200個体，20個体生存がデフォ

# 余談
LeakyESNというものもあるが今回は実装しない

# ga_dataの中身

  sp_acc.csv
  sp_acc1.csv
  sp_acc2.csv
  出来るだけたくさんの世代のバイアス無しの個体の精度を保存したもの，
  10世代に1回保存，tpの場合も同様
  weight.csv
  weight1.csv
  weight2.csv
  バイアス無し世代のReservoir層の重み，100世代に一回保存

  bias_sp_acc.csv
  出来るだけたくさんのバイアスあり，Heの初期化を用いた個体の精度を保存したもの
  10世代に1回保存，tpの場合も同様

  bias_weight_in.dot
  この時の入力層重みを保存したもの
  bias_weight.dot
  バイアス有り世代のReservoir層の重み，100世代に一回保存

  二つの入力層とReservoir層の重み二つを入れる事で途中から再開することが可能
  
  同上1.csv
  同上1.dat　等は同じ実行を別で行った物