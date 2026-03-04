# 🎨 Neural Style Transfer: Van Gogh Style
> **Gram Matrix** 기반의 스타일 전이 알고리즘을 활용하여 반고흐의 화풍을 풍경화에 합성하는 프로젝트입니다.

---

## 🧠 핵심 개념: Gram Matrix
스타일 전이의 핵심은 이미지의 '질감(Texture)'을 수학적으로 정의하는 것입니다. 본 프로젝트에서는 특정 레이어의 특징 맵(Feature Maps) 간의 상관관계를 계산하는 **Gram Matrix**를 사용했습니다.

### 1. 차원 변형 및 연산 과정 (B, C, H, W → B, C, C)
입력 데이터의 차원을 변경하여 스타일 특징을 추출하는 과정은 다음과 같습니다.

1.  **Input Shape**: `(Batch, Channels, Height, Width)`
2.  **Reshape**: 각 채널의 2D 특징들을 1D로 펼칩니다. 
    * `H * W`를 하나의 차원으로 합쳐 `(Batch, Channels, H * W)` 형태로 변형합니다.
3.  **Matrix Multiplication**: 변형된 행렬과 그 전치 행렬(Transpose)을 내적합니다.
    * $(C \times HW) \cdot (HW \times C) = (C \times C)$
4.  **Result**: 최종적으로 각 채널 간의 상관관계를 나타내는 `(Batch, Channels, Channels)` 형태의 Gram Matrix가 생성됩니다.



### 2. Mathematical Definition
$$G_{ij}^l = \sum_k F_{ik}^l F_{jk}^l$$
* $F_{ik}^l$: $l$번째 레이어의 $i$번째 필터가 위치 $k$에서 갖는 활성화 값.
* 이 연산을 통해 이미지의 구체적인 위치 정보는 사라지고, **어떤 특징(색상, 질감 등)이 동시에 자주 나타나는지**에 대한 스타일 정보만 남게 됩니다.

---

## 💻 Implementation (PyTorch)

```python
import torch

def get_gram_matrix(tensor):
    """
    Gram Matrix 계산 함수
    Input: (B, C, H, W)
    Output: (B, C, C)
    """
    b, c, h, w = tensor.size()
    
    # 1. B, C, H*W 형태로 펼치기 (Reshape)
    features = tensor.view(b, c, h * w)
    
    # 2. 전치 행렬과의 내적 (Batch-wise Matrix Multiplication)
    # features: (B, C, HW) / features.transpose: (B, HW, C)
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # 3. 값의 정규화 (Normalization)
    # 요소의 개수(C * H * W)로 나누어 scale을 조정합니다.
    return gram.div(c * h * w)
