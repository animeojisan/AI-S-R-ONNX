# AI Super-Resolution ONNX

AviUtl2 用の ONNX / WinML ベース AI フィルタープラグインです。

1x のデノイズ系モデルや 2～4x のアップスケーリングモデルを読み込み、AviUtl2 上で適用できます。

## 特徴

- AviUtl2 用 `.auf2` プラグイン
- ONNX ファイルをエクスプローラーから選択可能
- WinML を使用 (GPU/CPU対応)
- 1x / 2～4x モデル対応
- FP32 / FP16 モデル対応

## 使い方

1. `AI Super-Resolution ONNX.auf2` を AviUtl2 の `Plugin` フォルダに入れます
2. AviUtl2 を起動します
3. 動画または画像オブジェクトに **AI Super-Resolution ONNX** を追加します
4. `ONNXファイル` から使用する `.onnx` を選択します

## 注意

- 別オブジェクトとして置くのではなく、**既存オブジェクトのフィルタ効果として追加**してください
- 2～4x モデルは素材や使い方によって負荷が高くなることがあります
- モデルは付属していません。各自で用意してください
- SISR（Single Image Super-Resolution：単一画像超解像）　のみ対応。VSR　モデルは未対応
- ONNX モデルごとにライセンス条件が異なる場合があります


## ライセンス

本プラグイン本体のライセンスは別途同梱の `LICENSE` を参照してください。  
使用する ONNX モデルのライセンスは、各モデル配布元の条件に従ってください。
