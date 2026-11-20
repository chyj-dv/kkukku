# 제3자 고지 (Third-Party Notices)

꾸꾸(KKUKKU, 개발 이름 WearCast)는 아래 데이터와 모델을 사용합니다. 앱의 프로필 화면에도 같은 출처를 표시합니다.

## 학습 데이터: Fashion Product Images Dataset

* 출처: Param Aggarwal, Kaggle
  <https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset>
* 라이선스: MIT (Kaggle 데이터셋 페이지 표시 기준, 2026-11-10 확인)
* 사용 범위: 이 데이터셋의 의류 사진으로 카테고리 분류기(앱 안의 `assets/probes/*.json`)를
  학습했습니다. 분류기 파일에는 학습한 가중치만 들어 있고 이미지는 들어 있지 않습니다.
  데이터셋의 이미지와 파일은 이 저장소와 앱 배포물에 포함하지 않습니다.
* 상품 사진의 원 출처는 Myntra 이며, 사진에 대한 권리는 원 권리자에게 있습니다.

```
MIT License

Copyright (c) 2019 Param Aggarwal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 카테고리 제안 모델: Perception Encoder (PE-Core-S16)

* 출처: Meta FAIR, Daniel Bolya 외, "Perception Encoder: The best visual embeddings are not at the output of the network"
  (arXiv:2504.13181, 2025) <https://github.com/facebookresearch/perception_models>
* 가중치: OpenCLIP 형식 `PE-Core-S-16-384` <https://huggingface.co/timm/PE-Core-S-16-384>
* 라이선스: Apache License 2.0. 전문은 [licenses/PE-Core-Apache-2.0.txt](licenses/PE-Core-Apache-2.0.txt)
  (원 저장소의 `LICENSE.PE`, 2026-11-19 확인)
* 사용 범위와 바꾼 점: 이미지 인코더만 씁니다. 앱이 첫 실행 때 이 저장소의 Releases(`model-pe-core-s16-320-mixed`)에서
  `pe-core-s16-320-mixed.pte` 를 내려받습니다.
  * 입력 크기를 384×384 에서 320×320 으로 바꾸고 위치 임베딩을 보간했습니다.
  * 입력 정규화(평균·표준편차 0.5)를 모델 안에 넣었습니다.
  * ExecuTorch 형식(XNNPACK, fp16. 정규화 층만 float32)으로 변환했습니다.
  * 텍스트 인코더는 앱에 넣지 않았습니다. 카테고리·스타일 문장의 텍스트 임베딩을 PC 에서 미리 계산해
    앱 안의 `assets/models/pe-core-s16.labels.json` 에 넣었습니다.
  * 변환 코드는 kkukku-ml 저장소의 `experiments/exp1_model_selection/export_app_encoder.py` 입니다.

## 배경 제거 모델: U²-Net (u2netp)

* 출처: Xuebin Qin et al., U²-Net (Pattern Recognition 2020) <https://github.com/xuebinqin/U-2-Net>
* 라이선스: Apache License 2.0. 전문은 [licenses/U-2-Net-Apache-2.0.txt](licenses/U-2-Net-Apache-2.0.txt)
* 사용 범위: 경량 모델(u2netp)의 가중치를 rembg 가 배포하는 u2netp.onnx
  (<https://github.com/danielgatis/rembg>)에서 읽어, 직접 작성한 같은 구조의 모델에 옮긴 뒤
  ExecuTorch 형식(앱 안의 `assets/models/u2netp.pte`)으로 변환했습니다. 가중치 값은 바꾸지 않았고,
  배치 정규화는 원본 ONNX 처럼 합성곱에 합쳐진 상태입니다(kkukku-ml 저장소의 `experiments/exp3_segmentation/export_u2netp.py`).

## 날씨 데이터: Open-Meteo

* 출처: <https://open-meteo.com/>
* 라이선스: CC BY 4.0
