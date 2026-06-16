# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .event import EventNet
from .privacyNet import PrivacyNet


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,"EventNet":EventNet,"PrivacyNet":PrivacyNet}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
