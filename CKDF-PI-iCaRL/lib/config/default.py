from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "default"
_C.OUTPUT_DIR = "./output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (224, 224)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False


# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.dataset_name = "CIFAR100"        #mnist, mnist28, CIFAR10, CIFAR100, imagenet, svhn
_C.DATASET.dataset = "Torchvision_Datasets_Split"
_C.DATASET.data_root = "./datasets"
_C.DATASET.all_classes = 100
_C.DATASET.all_tasks = 10
_C.DATASET.split_seed = 0
_C.DATASET.val_length = 0
_C.DATASET.AUTO_ALPHA = True
_C.DATASET.use_svhn_extra = True


# cfg.exemplar_manager.memory_budget, cfg.exemplar_manager.mng_approach,
#                                        cfg.exemplar_manager.store_original_imgs, cfg.exemplar_manager.norm_exemplars,
#                                        cfg.exemplar_manager.centroid_order

# ----- exemplar_manager -----
_C.exemplar_manager = CN()
_C.exemplar_manager.store_original_imgs = True
_C.exemplar_manager.fixed_exemplar_num = -1
_C.exemplar_manager.memory_budget = 2000
_C.exemplar_manager.mng_approach = "herding"
_C.exemplar_manager.norm_exemplars = True
_C.exemplar_manager.centroid_order = "herding"

# ----- resume -----
_C.RESUME = CN()
_C.RESUME.use_resume = False
_C.RESUME.resumed_file = ""
_C.RESUME.resumed_model_path = ""
_C.RESUME.resumed_pre_tasks_model = ""

# ----- CLASSIFIER AUTO ALPHA -----
_C.AUTO_ALPHA = CN()
_C.AUTO_ALPHA.ALPHA = -1.
_C.AUTO_ALPHA.LENGTH = 100
_C.AUTO_ALPHA.GAMMA = 1.

_C.AUTO_ALPHA.LOSS0_FACTOR = 1.
_C.AUTO_ALPHA.LOSS1_FACTOR = 1.

# ----- BACKBONE BUILDER -----
_C.BACKBONE = CN()
_C.BACKBONE.TYPE = "resnext50"
_C.BACKBONE.PRETRAINED_BACKBONE = ""

# ----- MODULE BUILDER -----
_C.MODULE = CN()
_C.MODULE.TYPE = "GAP"

# ----- CLASSIFIER BUILDER -----
_C.CLASSIFIER = CN()
_C.CLASSIFIER.TYPE = "LDA"
_C.CLASSIFIER.BIAS = True

_C.CLASSIFIER.NECK = CN()
_C.CLASSIFIER.NECK.ENABLE = True
_C.CLASSIFIER.NECK.TYPE = 'Linear'
_C.CLASSIFIER.NECK.NUM_FEATURES = 2048
_C.CLASSIFIER.NECK.NUM_OUT = 128
_C.CLASSIFIER.NECK.HIDDEN_DIM = 512
_C.CLASSIFIER.NECK.MARGIN = 1.0
_C.CLASSIFIER.NECK.WEIGHT_INTER_LOSS = False
_C.CLASSIFIER.NECK.WEIGHT_INTRA_LOSS = False
_C.CLASSIFIER.NECK.INTER_DISTANCE = True
_C.CLASSIFIER.NECK.INTRA_DISTANCE = True
_C.CLASSIFIER.NECK.LOSS_FACTOR = 0.5

# ----- DISTILL -----
_C.DISTILL = CN()
_C.DISTILL.ENABLE = True
_C.DISTILL.CLS_DISTILL_ENABLE = False
_C.DISTILL.LOSS_FACTOR = 1.
_C.DISTILL.softmax_sigmoid = 0
_C.DISTILL.TEMP = 2.


# ----- LOSS BUILDER -----
_C.LOSS = CN()
_C.LOSS.LOSS_TYPE = "CrossEntropy"

# ----- TRAIN BUILDER -----
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.MAX_EPOCH = 90
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.TENSORBOARD = CN()
_C.TRAIN.TENSORBOARD.ENABLE = True
_C.TRAIN.SUM_GRAD = False


_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.TRAIN.OPTIMIZER.BASE_LR = 0.001
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5
_C.TRAIN.LR_SCHEDULER.COSINE_DECAY_END = 0


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
