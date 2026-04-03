# AI Super-Resolution ONNX

AviUtl2 用の ONNX / WinML ベース AI フィルタプラグインです。

1x のデノイズ系モデルや 2～4x のアップスケーリングモデルを読み込み、AviUtl2 上で適用できます。

## 特徴

- AviUtl2 用 `.auf2` プラグイン
- ONNX ファイルをエクスプローラーから選択可能
- WinML を使用 (GPU/CPU対応)
- 1x / 2～4x モデル対応
- FP32 / FP16 モデル対応

## 動作環境
- Windows環境
- AviUtl2
- ONNXモデルファイル
- GPUまたはCPUでの推論実行環境

※本プラグインは使用するモデルやPC環境により、動作速度・安定性・対応可否が異なります。

## 使い方

1. `AI Super-Resolution ONNX.auf2` を AviUtl2 の `Plugin` フォルダに入れます
2. AviUtl2 を起動します
3. 動画または画像オブジェクトに **AI Super-Resolution ONNX** を追加します
4. `ONNXファイル` から使用する `.onnx` を選択します

## 注意

- 別オブジェクトとして置くのではなく、**既存オブジェクトのフィルタ効果として追加**してください
- 2～4x モデルは素材や使い方によって負荷が高くなることがあります
- モデルは付属していません。各自で用意してください
- SISR（Single Image Super-Resolution：単一画像超解像） のみ対応。VSR モデルは未対応
- 対応モデルは float32 / float16 の 1入力1出力・4次元 NCHW 形式を想定しています
- 非対応モデルや読込失敗時は、フィルター効果が適用されない場合があります
- ONNX モデルごとにライセンス条件が異なる場合があります

## ONNX配布先

- [OpenModelDB](https://openmodeldb.info/)
- [hooke007](https://github.com/hooke007/dotfiles/releases/tag/onnx_models)
- [mpv-cHiDeNoise-AI](https://github.com/animeojisan/mpv-cHiDeNoise-AI) （\vs-plugins\models\55ai\ にあります）
- [anime4kOnnx](https://github.com/kato-megumi/anime4kOnnx)

## pth→onnx変換ツール

- [sisr2onnx](https://huggingface.co/spaces/Zarxrax/sisr2onnx)

## 免責事項
本ソフトウェアは個人制作によるものです。ご利用は各自の判断と責任でお願いいたします。
本ソフトウェアの導入、設定、使用により生じたいかなる不具合、損害、トラブル等についても作者は責任を負いかねます。


## ライセンス

本プラグイン本体のライセンスは別途同梱の `LICENSE` を参照してください。  
使用する ONNX モデルのライセンスは、各モデル配布元の条件に従ってください。
