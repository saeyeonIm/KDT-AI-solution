#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from yolox.exp import Exp as MyExp
import random
import numpy as np
import torch

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # 기본 설정
        self.cls_names = ['upper', 'bottoms', 'outwear', 'onepiece']
        self.depth = 0.33
        self.width = 0.375
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.seed = 16

        # 시드 설정
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # 모델 설정
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.num_classes = 4

        # 데이터 증강 설정 (중복 제거 및 최적화)
        self.mosaic_prob = 0.8      # 안정적인 학습을 위해 조정
        self.mixup_prob = 0.5       # 과적합 방지
        self.hsv_prob = 0.8         # 색상 변화
        self.flip_prob = 0.5        # 좌우 반전
        self.degrees = 10.0         # 회전 각도
        self.translate = 0.1        # 이동 범위
        self.mosaic_scale = (0.5, 1.5)  # 안정적인 스케일
        self.mixup_scale = (0.7, 1.3)   # 적절한 혼합 비율
        self.enable_mixup = True
        self.shear = 2.0           # 기하학적 변형

        # 학습 설정
        self.warmup_epochs = 5      # 안정적인 초기 학습
        self.max_epoch = 100        # 충분한 학습 시간
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.01 / 32.0  # tiny 모델에 최적화
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15     # 마지막 정리 단계
        self.ema = True             # 모델 평균화로 안정성 향상

        # 옵티마이저 설정
        self.weight_decay = 5e-4    # 과적합 방지
        self.momentum = 0.9
        self.data_num_workers = 4
        
        # 평가 및 로깅 설정
        self.eval_interval = 5      # 자주 평가
        self.print_interval = 20    # 적절한 로깅 간격
        self.save_interval = 5      # 모델 저장 간격

        # Early Stopping 설정
        self.early_stop_patience = 10
        self.early_stop_min_delta = 0.001
        self.best_ap = 0
        self.patience_counter = 0

    def after_eval(self, ap):
        """Early stopping 체크"""
        if ap > self.best_ap + self.early_stop_min_delta:
            self.best_ap = ap
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.early_stop_patience:
            print(f"Early stopping triggered. Best AP: {self.best_ap}")
            return True
        return False