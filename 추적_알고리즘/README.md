{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMwff7RFPXQ6apFkbRPpEJm",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/junsu122/AI_Application/blob/main/%EC%B6%94%EC%A0%81_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98/README.md\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 🎥 객체 추적 및 광학 흐름(Optical Flow) 기술 정리\n",
        "\n",
        "이미지 내 객체의 움직임을 파악하고 추적하기 위한 핵심 알고리즘들의 특징과 차이점을 정리합니다.\n",
        "\n",
        "---\n",
        "\n",
        "## 1. 컬러 기반 추적 (MeanShift & CamShift)\n",
        "\n",
        "### 🔹 MeanShift\n",
        "데이터의 밀도가 가장 높은 곳(무게 중심)을 찾아가는 알고리즘입니다.\n",
        "* **원리:** 타겟의 컬러 히스토그램을 기반으로, 다음 프레임에서 유사한 색상 분포를 가진 지역으로 윈도우를 이동시킵니다.\n",
        "* **특징:** 구현이 간단하지만, 객체의 크기가 변하면 대응하지 못합니다.\n",
        "\n",
        "### 🔹 CamShift (Continuously Adaptive MeanShift)\n",
        "MeanShift를 개선하여 객체의 크기와 회전 변화를 반영합니다.\n",
        "* **특징:** 객체가 카메라와 가까워지거나 멀어질 때 윈도우 크기를 스스로 조절합니다.\n",
        "* **용도:** 실시간 얼굴 추적이나 단순한 색상 물체 추적에 효과적입니다.\n",
        "\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "## 2. 특징점 기반 추적 (루카스-카나데, Lucas-Kanade)\n",
        "\n",
        "이미지 전체가 아닌 **특징점(Corner 등)**의 움직임만 계산하는 **희소(Sparse) 광학 흐름** 방식입니다.\n",
        "\n",
        "* **핵심 가정:** \"인접한 픽셀들은 움직임이 비슷하다\"는 가설을 바탕으로 연산합니다.\n",
        "* **장점:** 계산량이 적어 실시간 처리에 매우 유리합니다.\n",
        "* **단점:** 물체가 너무 빠르게 움직여 특징점이 범위를 벗어나면 추적에 실패할 수 있습니다. (이미지 피라미드 기법으로 보완 가능)\n",
        "\n",
        "---\n",
        "\n",
        "## 3. 딥러닝 기반 광학 흐름 (Deep Optical Flow & RAFT)\n",
        "\n",
        "신경망을 통해 이미지 내 **모든 픽셀의 움직임(Dense Flow)**을 정교하게 예측합니다.\n",
        "\n",
        "### 🔹 Deep Optical Flow (예: FlowNet)\n",
        "* 두 장의 이미지를 CNN에 입력하여 픽셀 단위의 이동량(Vector)을 출력합니다.\n",
        "* 고전적 방식보다 빛의 변화나 복잡한 배경에서 훨씬 강력합니다.\n",
        "\n",
        "### 🔹 RAFT (Recurrent All-Pairs Field Transforms)\n",
        "현재 SOTA(State-of-the-Art)급 성능을 보여주는 모델입니다.\n",
        "* **작동 방식:** 모든 픽셀 쌍의 유사도를 계산하고, GRU(순환 신경망)를 통해 반복적으로 흐름을 업데이트합니다.\n",
        "* **장점:** 아주 미세한 움직임부터 빠른 움직임까지 매우 정확하게 잡아내며 경계선 표현이 뛰어납니다.\n",
        "\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "## 📊 알고리즘 비교 요약\n",
        "\n",
        "| 구분 | 알고리즘 | 방식 | 주요 강점 |\n",
        "| :--- | :--- | :--- | :--- |\n",
        "| **컬러 기반** | MeanShift / CamShift | 히스토그램 밀도 추적 | 연산이 가볍고 색상 기반 추적에 유리 |\n",
        "| **특징점 기반** | 루카스-카나데 (L-K) | Sparse Optical Flow | 속도가 매우 빠름 (실시간성 최적) |\n",
        "| **딥러닝 기반** | RAFT / FlowNet | Dense Optical Flow | 복잡한 움직임 및 전 구역 정밀 분석 |\n",
        "\n",
        "---\n",
        "\n",
        "## 📊 한눈에 비교하기\n",
        "| 알고리즘 | 방식 | 주요 특징 | 추천 용도 |\n",
        "| :--- | :--- | :--- | :--- |\n",
        "| **MeanShift** | 컬러 기반 | 간단함, 윈도우 고정 | 일정한 크기의 색상 물체 추적 |\n",
        "| **CamShift** | 컬러 기반 | 크기/회전 적응형 | 크기 변화가 있는 객체 추적 |\n",
        "| **L-K Flow** | 특징점 기반 | 매우 빠름, Sparse | 실시간 특징점 추적, SLAM |\n",
        "| **RAFT** | 딥러닝 기반 | 매우 정교함, Dense | 고성능 영상 분석, 자율주행 |"
      ],
      "metadata": {
        "id": "0cN7doLnVTkQ"
      }
    }
  ]
}