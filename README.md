# SynviAI🚁

### Description

* __프로젝트명__: SynviAI

* __소속__: AIFFEL SeSAC, AIFFEL 대전

* __팀원__: 문경렬, 이건우, 이충열

* __담당퍼실__: 최해선(대전), 김국진(SeSAC)

* __프로젝트 개요__: 합성데이터를 학습하여 드론으로 촬영된 영상내 조난자 찾기

* __프로젝트 기간__: '21. 01 17 ~ 03. 08

* __기대하는 결과물__: 

  ​	Lv1. 학습용 데이터셋, 기본 모델 학습

  ​	Lv2. 데이터 증강에 따른 모델 성능 향상

  ​	Lv3. 모델 알고리즘 개선을 통한 성능 향상

  ​	Lv4. 실시간 영상처리를 통한 어플리케이션 개발

---

### 프로젝트 요약

* [아이템소개](#아이템-소개)
* [개발목적](#개발목적)
* [데이터 소개](#데이터-소개)
* [진행표](#진행표)
* [진행 세부사항](#진행-세부사항)
* [Yolo모델 코드분석](#Yolo모델-코드분석)

****

### 아이템 소개

* 드론으로 촬영된 영상내 조난자를 찾습니다.

* 제공되는 데이터의 증강 및 인공지능 모델의 개선을 통하여 드론으로 촬영한 영상에서 사람 검증합니다.

* 제공되는 데이터를 다양한 방식(프로그램 방식, 실 드론 데이터 획득 등)으로 증강합니다.

* 제공되는 AI기법을 Fine-Tuning하거나, 새로운 AI 기법을 통해 조난자(인물) 검출 AI모델 제안할 수 있습니다.

<p align="center"> <img src="https://github.com/chancho0518/AIFFELTHON/blob/main/image/person.png" height="300"/> </p>

----

### 개발목적

> __합성데이터__(Synthetic Data)
>
> 합성데이터(Synthetic Data)는 실제 또는 실험 데이터가 아닌 디지털 세계에서 프로그래밍 방식으로 생성되는 데이터로 딥러닝의 Classification의 성능을 높이기 위해 대용량 데이터를 확보하는 한편, 가상 3D 모델을 활용하여 실제 환경에서 얻기 힘든 데이터셋을 저비용으로 생성할 수 있습니다.  이미 여러 논문을 통해 실제 데이터보다 AI모델 훈련에 더 효율적이라는 사례가 있습니다. 

* 재난 상황에서 조난자를 구조하는데, AI가 도입되고 있고 이런 AI를 학습 시키기 위해 많은 데이터가 필요합니다. 

* 하지만 재난 데이터는 재현이 불가능하고. 데이터를 수집하는데 비용도 많이 발생합니다.
* 해당 프로젝트는 합성 데이터를 이용하여 AI의 데이터 이슈를 해결하는 것의 효용성을 증명하기 위함입니다.

* 합성 데이터로 재난 상황을 재현해서 데이터를 수집하고 이렇게 수집된 데이터로 재난 데이터가 필요한 정부기관과 AI연구기관에 데이터를 제공을 목적으로 합니다.

---

### 데이터 소개

* AI Hub 조난자 데이터 (https://aihub.or.kr/aidata/27687)

  * 드론 이동체 인지 영상(도로 고정)

  * 조난자 수색 부분, 이미지 30만장 총 800GB

  * 다운로드 및 사용을 위해 AIHub 가입과 데이터 사용 신청이 필요 (저작권 및 이용정책)

<p align="center"> <img src="https://github.com/chancho0518/AIFFELTHON/blob/main/image/image1.png" alt="image1.png" style="zoom:150%;" /> </p>



* Sim2Real 합성데이터
  * 10,216장, 총 30GB 
  * YOLO Dataset 기준 라벨링 정보

<p align="center"> <img src="https://github.com/chancho0518/AIFFELTHON/blob/main/image/image2.png" alt="image2.png" style="zoom:150%;" /> </p>



* Search and Rescue Dataset (SARD)

  * IEEE Dataport 데이터, 총 4GB

  * Json 라벨링 파일 → YOLO 라벨형식 제공

  * 안개효과 등 여러 필터효과를 통한 이미지 증강 사례

<p align="center"> <img src="https://github.com/chancho0518/AIFFELTHON/blob/main/image/image3.png" alt="image3.png" style="zoom:150%;" /> </p>

---

### 진행표
| Task                                               | 목표기간              | 세부내용                                                     |
| -------------------------------------------------- | --------------------- | ------------------------------------------------------------ |
| 데이터 EDA                                         | 2022.01.17~2022.01.21 | 합성데이터 전처리<br />실제데이터 전처리                     |
| Lv1: <br />학습용 데이터셋 기본 모델 학습          | 2022.01.24~2022.01.28 | YOLOv5<br />SSD<br />VTN                                     |
| Lv2:<br />데이터 증강에 따른 모델 성능 향상        | 2022.02.03~2022.02.11 | 데이터 증강<br />합성 데이터 생성                            |
| Lv3:<br />모델 알고리즘 개선을 통한 성능 향상      | 2022.02.14~2022.02.18 | 파인튜닝<br />알고리즘 개선                                  |
| Lv4:<br />실시간 영상처리를 통한 어플리케이션 개발 | 2022.02.21~2022.02.27 | 모델 알고리즘 개선<br />실시간 영상처리 <br />모델 고도화 TensorRT |
| 모델 최적화 및 디버깅                              | 2022.02.28~2022.03.08 | 모델 최적화 <br />디버깅 작업                                |
	
---

### 진행 세부사항

* Lv1 : 학습용 데이터셋, 기본 모델 학습
  * 합성데이터, 실제 조난자(사람) 데이터 수집 및 전처리
  * YOLOv5, SSD, VTN을 이용하여 조난자 디텍딩 확인
  * YOLO 모델은 큰이미지에서 작은 물체를 판별하는데 약점이 있어 SSD 모델을 함께 이용할 예정이며, YOLO와 SSD는 동영상 데이터에 적합하지 않아 VTN 모델을 함께 이용할 예정
* Lv2 : 데이터 증강에 따른 모델 성능 향상
  * 합성데이터, 공공데이터를 통한 데이터
  * 다양한 이미지 처리를 통한 데이터 증강기법 활용을 통한 정확도 확인
* Lv3 : 모델 알고리즘 개선을 통한 성능 향상
  * 모델 구조와 기법에 대한 이해 필요
* Lv4 : 실시간 영상처리를 통한 어플리케이션 개발
  * TensorRT 모델 최적화
  * 실시간 영상처리 시연, 이를 활용한 적절한 프로그램 개발

---

### Yolo모델 코드분석

* __바운딩박스 예측식__
<img src="https://github.com/chancho0518/AIFFELTHON/blob/main/image/func.png" alt="func.png" />


* __과정 1__: 욜로 배경이되는 darknet 모델 불러오기

* __과정 2__: 불러온 데이터를 딕셔너리로 처리

```python
#욜로 배경이되는 darknet 모델 불러오기

from __future__import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# 불러온 데이터를 딕셔너리로 처리합니다.
def create_modules(blocks):
    net_info = blocks[0] # 입력과 전처리에 대한 정보를 저장합니다.
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
		file = open(cfgfile, 'r')
	  lines = file.read().split('\n')               # lines를 list로 저장합니다.
	  lines = [x for x in lines if len(x) > 0]      # 빈 lines를 삭제합니다.
	  lines = [x for x in lines if x[0] != '#']     # 주석을 삭제합니다.
	  lines = [x.rstrip().lstrip() for x in lines]  # 공백을 제거합니다.

		block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':              # 새로운 block의 시작을 표시합니다.
            if len(block) != 0:         # block이 비어있지 않으면, 이전 block의 값을 저장합니다.
                blocks.append(block)    # 이것을 blocks list에 추가합니다.
                block = {}              # block을 초기화 합니다.
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks
```

* __과정 3__: 파이토치에 nn.Module class 를 사용하여 layer 확장

```python
def create_modules(blocks):
    net_info = blocks[0] # 입력과 전처리에 대한 정보를 저장합니다.
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
		for index, x in enumerate(blocks[1:]):
		    module = nn.Sequential()
		if (x['type'] == 'convolutional'):
            # layer에 대한 정보를 얻습니다.
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
								batch_normalize = 0
								bias = True
			
								filters = int(x['filters'])
								padding = int(x['pad'])
								kernel_size = int(x['size'])
								stride = int(x['stride'])
			
										if padding:
											pad = (kernel_size - 1) // 2
										else:
											pad = 0
			
								# convolutional layer를 추가합니다.
								conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
								module.add_module('conv_{0}'.format(index),conv)
								
								# Batch Norm Layer를 추가합니다.
								if batch_normalize:
									bn = nn.BatchNorm2d(filters)
									module.add_module('batch_norm_{0}'.format(index),bn)
									
								# activation을 확인합니다.
								# YOLO에서 Leaky ReLU 또는 Linear 입니다.
								if activation == 'leaky':
									activn = nn.LeakyReLU(0.1, inplace = True)
									module.add_module('leaky_{0}'.format(index), activn)
							
								# upsampling layer 입니다.
								# Bilinear2dUpsampling을 사용합니다.
			elif (x['type'] == 'upsample'):
				stride = int(x['stride'])
				upsample = nn.Upsample(sacle_factor = 2, mode = 'bilinear')
				module.add_module('upsample_{}'.format(index), upsample)
```

* __과정 4__: Route과 shortcut layer 를 추가

```python
# route layer 입니다.
elif (x['type'] == 'route'):
    x['layers'] = x['layers'].split(',')
    # route 시작
    start = int(x['layers'][0])
    # 1개만 존재하면 종료
    try:
        end = int(x['layers'][1])
        except:
            end = 0
            # 양수인 경우
            if start > 0:
                start = start - index
                if end > 0:
                    end = end - index
                    route = EmptyLayer()
                    module.add_module('route_{0}'.format(index), route)
                    # 음수 인 경우
                    if end < 0:
                        filters = output_filters[index + start] + output_filters[index + end]
                        else:
                            filters = output_filters[index + start]

                            # skip connection에 해당하는 shortcut
                            elif x['type'] == 'shortcut':
                                shortcut = EmptyLayer()
                                module.add_module('shortcut_{}'.format(index), shortcut)
```

