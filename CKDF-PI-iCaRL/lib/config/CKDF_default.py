from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "CKDF_default"
_C.OUTPUT_DIR = "./output"
_C.VALID_STEP = 20
_C.SAVE_STEP = 20
_C.SHOW_STEP = 100
_C.INPUT_SIZE = (32, 32)
_C.COLOR_SPACE = "RGB"
_C.CPU_MODE = False
_C.use_best_model = False
_C.availabel_cudas = ""
_C.use_base_half = False
_C.checkpoints = "./"
_C.task1_MODEL = ""
_C.save_model = False
_C.train_first_task = True

# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.dataset_name = "CIFAR100"  # mnist, mnist28, CIFAR10, CIFAR100, imagenet, svhn
_C.DATASET.dataset = "Torchvision_Datasets_Split"
_C.DATASET.data_json_file = ""
_C.DATASET.data_root = "./datasets"
_C.DATASET.all_classes = 100
_C.DATASET.all_tasks = 10
_C.DATASET.split_seed = 0
_C.DATASET.val_length = 0
_C.DATASET.use_svhn_extra = True

# ----- resume -----
_C.RESUME = CN()
_C.RESUME.use_resume = False
_C.RESUME.resumed_file = ""
_C.RESUME.resumed_model_path = ""

# ----- pre-train setting -----
_C.PRETRAINED = CN()
_C.PRETRAINED.use_pretrained_model = False
_C.PRETRAINED.MODEL = ""

# ----- exemplar_manager -----
_C.exemplar_manager = CN()
_C.exemplar_manager.store_original_imgs = True
_C.exemplar_manager.memory_budget = 2000
_C.exemplar_manager.mng_approach = "herding"
_C.exemplar_manager.norm_exemplars = True
_C.exemplar_manager.centroid_order = "herding"
_C.exemplar_manager.fixed_exemplar_num = -1
''' ----- BACKBONE BUILDER -----'''
_C.extractor = CN()
_C.extractor.TYPE = "resnet34"
_C.extractor.rate = 1.
_C.extractor.output_feature_dim = 512


#----- model -----'''
_C.model = CN()
#----- TRAIN -----'''
_C.model.TRAIN = CN()

_C.model.TRAIN.tradeoff_rate = 1.
_C.model.TRAIN.MAX_EPOCH = 120
_C.model.TRAIN.BATCH_SIZE = 128
_C.model.TRAIN.SHUFFLE = True
_C.model.TRAIN.NUM_WORKERS = 4
# ----- OPTIMIZER -----
_C.model.TRAIN.OPTIMIZER = CN()
_C.model.TRAIN.OPTIMIZER.BASE_LR = 0.1
_C.model.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.model.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.model.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4
# ----- LR_SCHEDULER -----
_C.model.TRAIN.LR_SCHEDULER = CN()
_C.model.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.model.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.model.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.model.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5

#----- FCTM_model -----'''
_C.FCTM = CN()
_C.FCTM.T = 2.
_C.FCTM.use_KD = True
_C.FCTM.use_binary_KD = True

# ----- FCN -----
_C.FCTM.use_FCN = True
_C.FCTM.FCN = CN()
_C.FCTM.FCN.in_feature_dim = 512
_C.FCTM.FCN.out_feature_dim = 512
_C.FCTM.FCN.layer_nums = 2
_C.FCTM.FCN.hidden_layer_rate = 2
_C.FCTM.FCN.last_hidden_layer_use_relu = False
#----- TRAIN -----'''
_C.FCTM.TRAIN = CN()
_C.FCTM.TRAIN.tradeoff_rate = 1.
_C.FCTM.TRAIN.MAX_EPOCH = 120
_C.FCTM.TRAIN.BATCH_SIZE = 128
_C.FCTM.TRAIN.SHUFFLE = True
_C.FCTM.TRAIN.NUM_WORKERS = 4
# ----- OPTIMIZER -----
_C.FCTM.TRAIN.OPTIMIZER = CN()
_C.FCTM.TRAIN.OPTIMIZER.BASE_LR = 0.1
_C.FCTM.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.FCTM.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.FCTM.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4
# ----- LR_SCHEDULER -----
_C.FCTM.TRAIN.LR_SCHEDULER = CN()
_C.FCTM.TRAIN.LR_SCHEDULER.TYPE = "multistep"
_C.FCTM.TRAIN.LR_SCHEDULER.LR_STEP = [30, 60]
_C.FCTM.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.1
_C.FCTM.TRAIN.LR_SCHEDULER.WARM_EPOCH = 5