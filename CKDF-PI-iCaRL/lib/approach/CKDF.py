import copy
import json
import os
import shutil
import time
import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from lib.data_transform.data_transform import AVAILABLE_TRANSFORMS
from lib.dataset import SubDataset
from lib.model import resnet_model, CrossEntropy, compute_distill_loss, CrossEntropy_binary, \
    compute_distill_binary_loss, FCN_model, compute_cls_distill_binary_loss
from lib.utils import AverageMeter, get_optimizer, TransformedDataset, ConcatDataset, TransformedDataset_for_exemplars, \
    get_scheduler, cuda_accuracy


class CKDF_handler:
    """Our approach DDC"""

    def __init__(self, dataset_handler, exemplar_manager, cfg, logger, device):
        self.dataset_handler = dataset_handler
        self.exemplar_manager = exemplar_manager
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.model = None
        self.pre_tasks_model = None

        self.FCN = None
        self.pre_FCN = None

        self.acc_result = None
        self.start_task_id = None

        self.latest_model = None
        self.best_model = None
        self.best_epoch = None
        self.best_acc = 0
        self.gpus = torch.cuda.device_count()
        self.device_ids = [] if cfg.availabel_cudas == "" else \
            [i for i in range(len(self.cfg.availabel_cudas.strip().split(',')))]

    def _first_task_init(self):
        '''Resume to init or init'''
        if self.cfg.RESUME.use_resume:
            self.logger.info(f"use_resume: {self.cfg.RESUME.resumed_model_path}")
            breakpoint_data = torch.load(self.cfg.RESUME.resumed_file)
            self.dataset_handler.update_split_selected_data(breakpoint_data["split_selected_data"])
            self.dataset_handler.get_dataset()
            self.resume_model()
            self.exemplar_manager.resume_manager(breakpoint_data)
            self.is_resume_legal()  # todo
            pass
        elif self.cfg.PRETRAINED.use_pretrained_model:
            self.dataset_handler.get_dataset()
            self.logger.info(f"use pretrained_model: {self.cfg.PRETRAINED_MODEL}")
            self.model = self.construct_model()
            self.logger.info(f"Pretrained extractor: {self.cfg.extractor.TYPE}, rate: {self.cfg.extractor.rate}")
            self.model.load_model(self.cfg.PRETRAINED_MODEL)
            if self.cfg.CPU_MODE:
                self.model = self.model.to(self.device)
            else:
                if self.cfg.CPU_MODE:
                    self.model = self.model.to(self.device)
                else:
                    if self.gpus > 1:
                        if len(self.device_ids) > 1:
                            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
                        else:
                            self.model = torch.nn.DataParallel(self.model).cuda()
                    else:
                        self.model = self.model.to("cuda")

        else:
            self.dataset_handler.get_dataset()
            self.model = self.construct_model()
            self.logger.info(f"extractor: {self.cfg.extractor.TYPE}, rate: {self.cfg.extractor.rate}")
            if self.cfg.CPU_MODE:
                self.model = self.model.to(self.device)
            else:
                if self.gpus > 1:
                    if len(self.device_ids) > 1:
                        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
                    else:
                        self.model = torch.nn.DataParallel(self.model).cuda()
                else:
                    self.model = self.model.to("cuda")

    def resume_model(self):
        # todo done!
        checkpoint = torch.load(self.cfg.RESUME.resumed_model_path)
        self.acc_result = checkpoint['acc_result']
        self.start_task_id = checkpoint['task_id']
        self.model = self.construct_model()
        self.model.load_model(self.cfg.RESUME.resumed_model_path)
        self.logger.info(f"Resumed extractor: {self.cfg.extractor.TYPE}, rate: {self.cfg.extractor.rate}")
        if self.start_task_id > 1 or (self.cfg.use_base_half and
                                      self.start_task_id > int(self.dataset_handler.all_tasks / 2)):
            self.FCN = FCN_model(self.cfg)
        if self.cfg.CPU_MODE:
            self.model = self.model.to(self.device)
            if self.FCN is not None:
                self.FCN = self.FCN.to(self.device)
        else:
            if self.gpus > 1:
                if len(self.device_ids) > 1:
                    self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
                    if self.FCN is not None:
                        self.FCN = torch.nn.DataParallel(self.FCN, device_ids=self.device_ids).cuda()
                else:
                    self.model = torch.nn.DataParallel(self.model).cuda()
                    if self.FCN is not None:
                        self.FCN = torch.nn.DataParallel(self.FCN).cuda()
            else:
                self.model = self.model.to("cuda")
                if self.FCN is not None:
                    self.FCN = self.FCN.cuda()
        self.logger.info(f"start from task {self.start_task_id}")
        pass

    def construct_model(self):
        model = resnet_model(self.cfg)
        return model
        pass

    def is_resume_legal(self):
        # todo
        learned_classes_num = len(self.exemplar_manager.exemplar_sets)
        assert learned_classes_num % self.dataset_handler.classes_per_task == 0
        assert learned_classes_num / self.dataset_handler.classes_per_task == self.start_task_id
        self.logger.info(f"Resume acc_result of resumed model: {self.acc_result}")
        acc = self.validate_with_exemplars(task=self.start_task_id)
        FC_acc = self.validate_with_FC(task=self.start_task_id)
        self.logger.info(f"validate resumed model: {acc.mean()} || {FC_acc.mean()}")
        taskIL_acc = self.validate_with_exemplars_taskIL(self.start_task_id)
        taskIL_FC_acc = self.validate_with_FC_taskIL(self.start_task_id)
        self.logger.info(f"validate resumed model: {taskIL_acc.mean()} || {taskIL_FC_acc.mean()}")
        pass

    def build_optimize(self, model=None, base_lr=None, optimizer_type=None, momentum=None, weight_decay=None):
        # todo Done
        MODEL = model if model else self.model
        optimizer = get_optimizer(MODEL, BASE_LR=base_lr, optimizer_type=optimizer_type, momentum=momentum,
                                  weight_decay=weight_decay)

        return optimizer

    def build_scheduler(self, optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None, MAX_EPOCH=200):
        # todo optimizer, lr_type=None, lr_step=None, lr_factor=None, warmup_epochs=None
        scheduler = get_scheduler(optimizer=optimizer, lr_type=lr_type, lr_step=lr_step, lr_factor=lr_factor,
                                  warmup_epochs=warmup_epochs, MAX_EPOCH=MAX_EPOCH)
        return scheduler

    def CKDF_train_main(self):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        self.logger.info(f"use {self.gpus} gpus")
        cudnn.benchmark = True
        cudnn.enabled = True
        # 初始化 Network
        self._first_task_init()
        print(self.model)
        train_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
        ])

        if not self.cfg.RESUME.use_resume:
            self.start_task_id = 1  # self.start_task_id 从 1 开始
        else:
            self.start_task_id += 1

        model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")
        code_dir = os.path.join(self.cfg.OUTPUT_DIR, "codes")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            self.logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )

            shutil.rmtree(code_dir)
        self.logger.info("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        shutil.copytree("/n02dat01/users/kcli/CL_research/CKDF-PI-iCaRL", code_dir, ignore=ignore)
        train_dataset = None
        train_dataset_for_EM = None
        for task, original_imgs_train_dataset in enumerate(self.dataset_handler.original_imgs_train_datasets,
                                                           1):
            self.logger.info(f'New task {task} begin.')

            if self.cfg.RESUME.use_resume and task < self.start_task_id:
                self.logger.info(f"Use resume. continue.")
                continue

            if self.cfg.use_base_half and task < int(self.dataset_handler.all_tasks / 2):
                train_dataset_temp = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

                if train_dataset is None:
                    train_dataset = train_dataset_temp
                else:
                    train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                if self.cfg.exemplar_manager.memory_budget > 0:
                    if self.cfg.exemplar_manager.store_original_imgs:
                        train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                     transform=
                                                                                     self.dataset_handler.val_test_dataset_transform)
                    else:
                        train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                     transform=train_dataset_transform)

                    if train_dataset_for_EM is None:
                        train_dataset_for_EM = train_dataset_for_EM_temp
                    else:
                        train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
                self.logger.info(f'task continue.')
                continue
            else:
                if self.cfg.use_base_half:
                    if task == int(self.dataset_handler.all_tasks / 2):
                        train_dataset_temp = TransformedDataset(original_imgs_train_dataset,
                                                                transform=train_dataset_transform)

                        train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                        self.logger.info(f'base_half dataset construct end.')
                        # self.batch_train_logger.info(f'base_half dataset construct end.')
                        self.logger.info(f'train_dataset length: {len(train_dataset)}.')
                    elif task > int(self.dataset_handler.all_tasks / 2):
                        train_dataset = TransformedDataset(original_imgs_train_dataset,
                                                           transform=train_dataset_transform)
                    else:
                        train_dataset = None
                else:
                    train_dataset = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

            exemplar_dataset = None
            if self.exemplar_manager.memory_budget > 0:
                if self.cfg.use_base_half:
                    if task > int(self.dataset_handler.all_tasks / 2):
                        exemplar_dataset = self.exemplar_manager.get_ExemplarDataset(for_train=True)

                elif task > 1:
                    exemplar_dataset = self.exemplar_manager.get_ExemplarDataset(for_train=True)

            if task > 1:
                if exemplar_dataset:
                    self.logger.info(f"exemplar_dataset length: {len(exemplar_dataset)} ")
                else:
                    self.logger.info(f"exemplar_dataset length: None ")

            # Find [active_classes]
            active_classes_num = self.dataset_handler.classes_per_task * task
            if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) or \
                    (not self.cfg.use_base_half and task == 1):
                if self.cfg.train_first_task:
                    self.first_task_train_main(train_dataset, active_classes_num, task)
                else:
                    self.model = self.construct_model()
                    self.model.load_model(self.cfg.task1_MODEL)
                    if self.gpus > 1:
                        if len(self.device_ids) > 1:
                            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
                        else:
                            self.model = torch.nn.DataParallel(self.model).cuda()
                    else:
                        self.model = self.model.to("cuda")
            else:
                self.pre_tasks_model = copy.deepcopy(self.model).eval()
                if self.FCN is None:
                    self.FCN = FCN_model(self.cfg)
                    print(self.FCN)
                    if self.gpus > 1:
                        if len(self.device_ids) > 1:
                            self.FCN = torch.nn.DataParallel(self.FCN, device_ids=self.device_ids).cuda()
                        else:
                            self.FCN = torch.nn.DataParallel(self.FCN).cuda()
                    else:
                        self.FCN = self.FCN.cuda()
                if self.cfg.exemplar_manager.memory_budget > 0:
                    training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
                else:
                    training_dataset = train_dataset
                self.logger.info(f'Task {task} begin: train FCTM_train_main.')
                self.FCN_train_main(training_dataset, active_classes_num, task)  # todo Done!

                self.pre_FCN = copy.deepcopy(self.FCN).eval()
                self.logger.info(f'Task {task} train model begin:')
                self.train_main(training_dataset, active_classes_num, task)  # todo Done!

            self.logger.info(f'#############MCFM train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')
            # print("DDC train task-%d End" % task)
            # print("Example handler task-%d start." % task)
            # EXEMPLARS: update exemplar sets
            self.latest_model = self.model
            if self.cfg.use_best_model:
                self.model = self.best_model
                self.logger.info(f"Use best model. ")
            if self.cfg.save_model:
                torch.save({
                    'state_dict': self.latest_model.state_dict(),
                    'task_id': task
                }, os.path.join(model_dir, "latest_model.pth")
                )
            if self.cfg.exemplar_manager.memory_budget > 0:
                exemplars_per_class_list = []
                if self.cfg.exemplar_manager.fixed_exemplar_num > 0:
                    for class_id in range(active_classes_num):
                        exemplars_per_class_list.append(self.cfg.exemplar_manager.fixed_exemplar_num)
                else:
                    exemplars_per_class = int(np.floor(self.exemplar_manager.memory_budget / active_classes_num))
                    delta_size = self.exemplar_manager.memory_budget % active_classes_num
                    for class_id in range(active_classes_num):
                        if delta_size > 0:
                            exemplars_per_class_list.append(exemplars_per_class + 1)
                            delta_size -= 1
                        else:
                            exemplars_per_class_list.append(exemplars_per_class)
                # reduce examplar-sets
                if self.cfg.exemplar_manager.fixed_exemplar_num < 0:
                    if self.cfg.use_base_half and task > int(self.dataset_handler.all_tasks / 2) or \
                            (not self.cfg.use_base_half and task > 1):
                        self.exemplar_manager.reduce_exemplar_sets(exemplars_per_class_list)

                if self.cfg.exemplar_manager.store_original_imgs:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=
                                                                                 self.dataset_handler.val_test_dataset_transform)
                else:
                    train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                 transform=train_dataset_transform)
                # for each new class trained on, construct examplar-set
                if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2):
                    new_classes = list(range(0, self.dataset_handler.classes_per_task * task))
                    train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
                else:
                    new_classes = list(range(self.dataset_handler.classes_per_task * (task - 1),
                                             self.dataset_handler.classes_per_task * task))
                    train_dataset_for_EM = train_dataset_for_EM_temp

                for class_id in new_classes:
                    # create new dataset containing only all examples of this class
                    self.logger.info(f"construct_exemplar_set class_id: {class_id}")
                    class_dataset = SubDataset(original_dataset=train_dataset_for_EM,
                                               sub_labels=[class_id])
                    # based on this dataset, construct new exemplar-set for this class
                    self.exemplar_manager.construct_exemplar_set(class_dataset, self.model,
                                                                 exemplars_per_class_list[class_id],
                                                                 self.cfg.model.TRAIN.BATCH_SIZE,
                                                                 self.cfg.model.TRAIN.NUM_WORKERS,
                                                                 feature_flag=True)
                    self.logger.info(
                        f"self.exemplar_manager exemplar_set length: {len(self.exemplar_manager.exemplar_sets)}")
                self.exemplar_manager.compute_means = True
                self.exemplar_manager.recompute_centroid_feature = True
                val_acc_with_exemplars_ncm = self.validate_with_exemplars(task)
                taskIL_val_acc = self.validate_with_exemplars_taskIL(task)

            val_acc = self.validate_with_FC(task=task)
            taskIL_FC_val_acc = self.validate_with_FC_taskIL(task=task)
            test_acc = None
            if self.dataset_handler.val_datasets:
                if self.cfg.exemplar_manager.memory_budget > 0:
                    test_acc_with_exemplars_ncm = self.validate_with_exemplars(task=task, is_test=True)
                    taskIL_test_acc = self.validate_with_exemplars_taskIL(task=task, is_test=True)
                test_acc = self.validate_with_FC(task=task, is_test=True)
                taskIL_FC_test_acc = self.validate_with_FC_taskIL(task=task, is_test=True)
            if test_acc:
                if self.cfg.save_model:
                    self.save_best_latest_model_data(model_dir, task, test_acc.mean(), self.cfg.model.TRAIN.MAX_EPOCH)
            else:
                if self.cfg.save_model:
                    self.save_best_latest_model_data(model_dir, task, val_acc.mean(), self.cfg.model.TRAIN.MAX_EPOCH)
            '''if test_acc:
                self.save_best_latest_model_data(model_dir, task, test_acc.mean(), self.cfg.TRAIN.MAX_EPOCH)
            else:
                self.save_best_latest_model_data(model_dir, task, val_acc.mean(), self.cfg.TRAIN.MAX_EPOCH)'''
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            if self.dataset_handler.val_datasets:
                val_acc_FC_str = f'task: {task} classififer:{"FC"} val_acc: {val_acc}, avg: {val_acc.mean()} '
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {test_acc}, avg: {test_acc.mean()} '
                self.logger.info(val_acc_FC_str)
                self.logger.info(test_acc_FC_str)
                if self.cfg.exemplar_manager.memory_budget > 0:
                    val_acc_ncm_str = f'task: {task} classififer:{"ncm"} val_acc: {val_acc_with_exemplars_ncm}, ' \
                                      f'avg: {val_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    test_acc_ncm_str = f'task: {task} classififer:{"ncm"} test_acc: {test_acc_with_exemplars_ncm}, ' \
                                       f'avg: {test_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    self.logger.info(val_acc_ncm_str)
                    self.logger.info(test_acc_ncm_str)
                    self.logger.info(
                        f"validate taskIL: NCM: {taskIL_test_acc.mean()} || FC: {taskIL_FC_test_acc.mean()}")
                else:
                    self.logger.info(f"validate taskIL: FC: {taskIL_FC_test_acc.mean()}")
            else:
                if self.cfg.exemplar_manager.memory_budget > 0:
                    test_acc_ncm_str = f'task: {task} classififer:{"ncm"} test_acc: {val_acc_with_exemplars_ncm}, ' \
                                       f'avg: {val_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    self.logger.info(test_acc_ncm_str)
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {val_acc}, avg: {val_acc.mean()} '
                self.logger.info(test_acc_FC_str)
                # print(f"validate resumed model: {taskIL_acc.mean()} || {taskIL_FC_acc.mean()}")
                if self.cfg.exemplar_manager.memory_budget > 0:
                    self.logger.info(f"validate taskIL: NCM: {taskIL_val_acc} || FC: {taskIL_FC_val_acc}")
                    self.logger.info(f"validate taskIL: NCM: {taskIL_val_acc.mean()} || FC: {taskIL_FC_val_acc.mean()}")
                else:
                    self.logger.info(f"validate taskIL: FC: {taskIL_FC_val_acc.mean()}")

    def CKDF_train_main_for_local_dataset(self):
        '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

        [train_datasets]    <list> with for each task the training <DataSet>
        [scenario]          <str>, choice from "task", "domain" and "class"
        [classes_per_task]  <int>, # of classes per task'''
        gpus = torch.cuda.device_count()
        self.logger.info(f"use {gpus} gpus")
        cudnn.benchmark = True
        cudnn.enabled = True
        # 初始化 Network
        self._first_task_init()
        print(self.model)
        train_dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[self.dataset_handler.dataset_name]['train_transform'],
        ])

        if not self.cfg.RESUME.use_resume:
            self.start_task_id = 1  # self.start_task_id 从 1 开始
        else:
            self.start_task_id += 1

        model_dir = os.path.join(self.cfg.OUTPUT_DIR, "models")
        code_dir = os.path.join(self.cfg.OUTPUT_DIR, "codes")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            self.logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )

            shutil.rmtree(code_dir)
        self.logger.info("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )
        shutil.copytree("/n02dat01/users/kcli/CL_research/CKDF-PI-iCaRL", code_dir, ignore=ignore)
        train_dataset = None
        train_dataset_for_EM = None
        for task, original_imgs_train_dataset in enumerate(self.dataset_handler.original_imgs_train_datasets,
                                                           1):
            self.logger.info(f'New task {task} begin.')

            if self.cfg.RESUME.use_resume and task < self.start_task_id:
                self.logger.info(f"Use resume. continue.")
                continue

            if self.cfg.use_base_half and task < int(self.dataset_handler.all_tasks / 2):
                train_dataset_temp = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

                if train_dataset is None:
                    train_dataset = train_dataset_temp
                else:
                    train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                if self.cfg.exemplar_manager.memory_budget > 0:
                    if self.cfg.exemplar_manager.store_original_imgs:
                        train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                     transform=
                                                                                     self.dataset_handler.val_test_dataset_transform)
                    else:
                        train_dataset_for_EM_temp = TransformedDataset_for_exemplars(original_imgs_train_dataset,
                                                                                     transform=train_dataset_transform)

                    if train_dataset_for_EM is None:
                        train_dataset_for_EM = train_dataset_for_EM_temp
                    else:
                        train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
                self.logger.info(f'task continue.')
                continue
            else:
                if self.cfg.use_base_half:
                    if task == int(self.dataset_handler.all_tasks / 2):
                        train_dataset_temp = TransformedDataset(original_imgs_train_dataset,
                                                                transform=train_dataset_transform)

                        train_dataset = ConcatDataset([train_dataset, train_dataset_temp])
                        self.logger.info(f'base_half dataset construct end.')
                        # self.batch_train_logger.info(f'base_half dataset construct end.')
                        self.logger.info(f'train_dataset length: {len(train_dataset)}.')
                    elif task > int(self.dataset_handler.all_tasks / 2):
                        train_dataset = TransformedDataset(original_imgs_train_dataset,
                                                           transform=train_dataset_transform)
                    else:
                        train_dataset = None
                else:
                    train_dataset = TransformedDataset(original_imgs_train_dataset, transform=train_dataset_transform)

            exemplar_dataset = None
            if self.exemplar_manager.memory_budget > 0:
                if self.cfg.use_base_half:
                    if task > int(self.dataset_handler.all_tasks / 2):
                        exemplar_dataset = self.exemplar_manager.get_ExemplarDataset(for_train=True)

                elif task > 1:
                    exemplar_dataset = self.exemplar_manager.get_ExemplarDataset(for_train=True)

            if task > 1:
                if exemplar_dataset:
                    self.logger.info(f"exemplar_dataset length: {len(exemplar_dataset)} ")
                else:
                    self.logger.info(f"exemplar_dataset length: None ")

            # Find [active_classes]
            active_classes_num = self.dataset_handler.classes_per_task * task
            if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2) or \
                    (not self.cfg.use_base_half and task == 1):
                if self.cfg.train_first_task:
                    self.first_task_train_main(train_dataset, active_classes_num, task)
                else:
                    self.model = self.construct_model()
                    self.model.load_model(self.cfg.task1_MODEL)
                    device_ids = [i for i in range(len(self.cfg.availabel_cudas.strip().split(',')))]
                    if len(device_ids) > 1:
                        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids).cuda()
                    else:
                        self.model = self.model.cuda()
            else:
                self.pre_tasks_model = copy.deepcopy(self.model).eval()
                if self.FCN is None:
                    self.FCN = FCN_model(self.cfg)
                    print(self.FCN)
                    device_ids = [i for i in range(len(self.cfg.availabel_cudas.strip().split(',')))]
                    if len(device_ids) > 1:
                        self.FCN = torch.nn.DataParallel(self.FCN, device_ids=device_ids).cuda()
                    else:
                        self.FCN = self.FCN.cuda()

                if self.cfg.exemplar_manager.memory_budget > 0:
                    training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
                else:
                    training_dataset = train_dataset
                self.logger.info(f'Task {task} begin: train FCTM_train_main.')
                self.FCN_train_main(training_dataset, active_classes_num, task)  # todo Done!

                self.pre_FCN = copy.deepcopy(self.FCN).eval()
                self.logger.info(f'Task {task} train model begin:')
                self.train_main(training_dataset, active_classes_num, task)  # todo Done!
            self.logger.info(f'#############MCFM train task {task} End.##############')
            self.logger.info(f'#############Example handler task {task} start.##############')
            # print("DDC train task-%d End" % task)
            # print("Example handler task-%d start." % task)
            # EXEMPLARS: update exemplar sets
            self.latest_model = self.model
            if self.cfg.use_best_model:
                self.model = self.best_model
                self.logger.info(f"Use best model. ")
            if self.cfg.save_model:
                torch.save({
                    'state_dict': self.latest_model.state_dict(),
                    'task_id': task
                }, os.path.join(model_dir, "latest_model.pth")
                )
            if self.cfg.exemplar_manager.memory_budget > 0:
                exemplars_per_class_list = []
                if self.cfg.exemplar_manager.fixed_exemplar_num > 0:
                    for class_id in range(active_classes_num):
                        exemplars_per_class_list.append(self.cfg.exemplar_manager.fixed_exemplar_num)
                else:
                    exemplars_per_class = int(np.floor(self.exemplar_manager.memory_budget / active_classes_num))
                    delta_size = self.exemplar_manager.memory_budget % active_classes_num
                    for class_id in range(active_classes_num):
                        if delta_size > 0:
                            exemplars_per_class_list.append(exemplars_per_class + 1)
                            delta_size -= 1
                        else:
                            exemplars_per_class_list.append(exemplars_per_class)
                # reduce examplar-sets
                if self.cfg.exemplar_manager.fixed_exemplar_num < 0:
                    if self.cfg.use_base_half and task > int(self.dataset_handler.all_tasks / 2) or \
                            (not self.cfg.use_base_half and task > 1):
                        self.exemplar_manager.reduce_exemplar_sets(exemplars_per_class_list)

                if self.cfg.use_base_half and task == int(self.dataset_handler.all_tasks / 2):
                    new_classes = list(range(0, self.dataset_handler.classes_per_task * task))
                    # train_dataset_for_EM = ConcatDataset([train_dataset_for_EM, train_dataset_for_EM_temp])
                    train_dataset_for_EM.extend(self.dataset_handler.original_imgs_train_datasets_per_class[task - 1])
                else:
                    new_classes = list(range(self.dataset_handler.classes_per_task * (task - 1),
                                             self.dataset_handler.classes_per_task * task))
                    train_dataset_for_EM = self.dataset_handler.original_imgs_train_datasets_per_class[task - 1]
                assert len(new_classes) == len(train_dataset_for_EM)
                for class_id in range(len(new_classes)):
                    # create new dataset containing only all examples of this class
                    self.logger.info(f"construct_exemplar_set class_id: {new_classes[class_id]}")

                    if self.cfg.exemplar_manager.store_original_imgs:
                        class_dataset = TransformedDataset_for_exemplars(train_dataset_for_EM[class_id],
                                                                         store_original_imgs=self.cfg.exemplar_manager.store_original_imgs,
                                                                         transform=
                                                                         self.dataset_handler.val_test_dataset_transform)
                    else:
                        class_dataset = TransformedDataset_for_exemplars(train_dataset_for_EM[class_id],
                                                                         store_original_imgs=self.cfg.exemplar_manager.store_original_imgs,
                                                                         transform=train_dataset_transform)
                    # based on this dataset, construct new exemplar-set for this class
                    self.exemplar_manager.construct_exemplar_set(class_dataset, self.model,
                                                                 exemplars_per_class_list[new_classes[class_id]],
                                                                 self.cfg.model.TRAIN.BATCH_SIZE,
                                                                 self.cfg.model.TRAIN.NUM_WORKERS,
                                                                 feature_flag=True)
                    self.logger.info(
                        f"self.exemplar_manager exemplar_set length: {len(self.exemplar_manager.exemplar_sets)}")
                self.exemplar_manager.compute_means = True
                self.exemplar_manager.recompute_centroid_feature = True
                val_acc_with_exemplars_ncm = self.validate_with_exemplars(task)
                taskIL_val_acc = self.validate_with_exemplars_taskIL(task)

            val_acc = self.validate_with_FC(task=task)
            taskIL_FC_val_acc = self.validate_with_FC_taskIL(task=task)
            test_acc = None
            if self.dataset_handler.val_datasets:
                if self.cfg.exemplar_manager.memory_budget > 0:
                    test_acc_with_exemplars_ncm = self.validate_with_exemplars(task=task, is_test=True)
                    taskIL_test_acc = self.validate_with_exemplars_taskIL(task=task, is_test=True)
                test_acc = self.validate_with_FC(task=task, is_test=True)
                taskIL_FC_test_acc = self.validate_with_FC_taskIL(task=task, is_test=True)
            if test_acc:
                if self.cfg.save_model:
                    self.save_best_latest_model_data(model_dir, task, test_acc.mean(), self.cfg.model.TRAIN.MAX_EPOCH)
            else:
                if self.cfg.save_model:
                    self.save_best_latest_model_data(model_dir, task, val_acc.mean(), self.cfg.model.TRAIN.MAX_EPOCH)
            '''if test_acc:
                self.save_best_latest_model_data(model_dir, task, test_acc.mean(), self.cfg.TRAIN.MAX_EPOCH)
            else:
                self.save_best_latest_model_data(model_dir, task, val_acc.mean(), self.cfg.TRAIN.MAX_EPOCH)'''
            self.logger.info(f'#############task: {task:0>3d} is finished Test begin. ##############')
            if self.dataset_handler.val_datasets:
                val_acc_FC_str = f'task: {task} classififer:{"FC"} val_acc: {val_acc}, avg: {val_acc.mean()} '
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {test_acc}, avg: {test_acc.mean()} '
                self.logger.info(val_acc_FC_str)
                self.logger.info(test_acc_FC_str)
                if self.cfg.exemplar_manager.memory_budget > 0:
                    val_acc_ncm_str = f'task: {task} classififer:{"ncm"} val_acc: {val_acc_with_exemplars_ncm}, ' \
                                      f'avg: {val_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    test_acc_ncm_str = f'task: {task} classififer:{"ncm"} test_acc: {test_acc_with_exemplars_ncm}, ' \
                                       f'avg: {test_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    self.logger.info(val_acc_ncm_str)
                    self.logger.info(test_acc_ncm_str)
                    self.logger.info(
                        f"validate taskIL: NCM: {taskIL_test_acc.mean()} || FC: {taskIL_FC_test_acc.mean()}")
                else:
                    self.logger.info(f"validate taskIL: FC: {taskIL_FC_test_acc.mean()}")
            else:
                if self.cfg.exemplar_manager.memory_budget > 0:
                    test_acc_ncm_str = f'task: {task} classififer:{"ncm"} test_acc: {val_acc_with_exemplars_ncm}, ' \
                                       f'avg: {val_acc_with_exemplars_ncm.mean()}, classififer:{"centroid"} '
                    self.logger.info(test_acc_ncm_str)
                test_acc_FC_str = f'task: {task} classififer:{"FC"} || test_acc: {val_acc}, avg: {val_acc.mean()} '
                self.logger.info(test_acc_FC_str)
                # print(f"validate resumed model: {taskIL_acc.mean()} || {taskIL_FC_acc.mean()}")
                if self.cfg.exemplar_manager.memory_budget > 0:
                    self.logger.info(f"validate taskIL: NCM: {taskIL_val_acc} || FC: {taskIL_FC_val_acc}")
                    self.logger.info(f"validate taskIL: NCM: {taskIL_val_acc.mean()} || FC: {taskIL_FC_val_acc.mean()}")
                else:
                    self.logger.info(f"validate taskIL: FC: {taskIL_FC_val_acc.mean()}")

    def FCN_train_main(self, train_dataset, active_classes_num, task):
        optimizer = self.build_optimize(model=self.FCN,
                                        base_lr=self.cfg.FCTM.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.FCTM.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.FCTM.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.FCTM.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.FCTM.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.FCTM.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.FCTM.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.FCTM.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.FCTM.TRAIN.MAX_EPOCH)
        MAX_EPOCH = self.cfg.FCTM.TRAIN.MAX_EPOCH
        best_acc = 0
        all_cls_criterion = CrossEntropy().cuda()
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.FCTM.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.FCTM.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        for epoch in range(1, MAX_EPOCH + 1):
            all_loss = AverageMeter()
            acc = AverageMeter()
            iter_index = 0
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            for x, y in train_loader:
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                loss, now_acc, now_cnt = self.FCN_train_a_batch(optimizer, all_cls_criterion, x, y,
                                                                active_classes_num,
                                                                self.dataset_handler.classes_per_task)
                all_loss.update(loss.data.item(), cnt)
                acc.update(now_acc, y.shape[0])
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "FCTM train, Epoch: {} || Batch:{:>3d}/{} || lr : {} || Loss:{:>5.3f} || " \
                               "Accuracy:{:>5.2f}".format(epoch,
                                                          iter_index,
                                                          iter_num,
                                                          optimizer.param_groups[
                                                              0]['lr'],
                                                          all_loss.val,
                                                          acc.val * 100,
                                                          )
                    self.logger.info(pbar_str)
                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FCTM(task)  # task_id 从1开始
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------FCTM train Epoch:{:>3d}    FCTM val_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------FCTM train Best_Epoch:{:>3d}   FCTM Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        pass

    def construct_weight_per_class(self, active_classes_num, current_task_classes_imgs_num, beta=0.95):
        cls_num_list = [len(self.exemplar_manager.exemplar_sets[0])] * \
                       (active_classes_num - self.dataset_handler.classes_per_task) + [
                           current_task_classes_imgs_num for i in range(self.dataset_handler.classes_per_task)]

        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / \
                          np.sum(per_cls_weights) * len(cls_num_list)

        self.logger.info("per cls weights : {}".format(per_cls_weights))
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        return per_cls_weights

    def construct_sample_num_per_class(self, active_classes_num, current_task_classes_imgs_num):
        pre_task_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        sample_num_per_class = np.array([0, ] * active_classes_num)
        assert len(self.exemplar_manager.exemplar_sets) == pre_task_classes_num
        for i in range(len(self.exemplar_manager.exemplar_sets)):
            sample_num_per_class[i] = len(self.exemplar_manager.exemplar_sets[i])
        sample_num_per_class[pre_task_classes_num:active_classes_num] = current_task_classes_imgs_num
        return torch.from_numpy(sample_num_per_class).float()

    def FCN_train_a_batch(self, optimizer, all_cls_criterion, imgs, labels, active_classes_num,
                          classes_per_task):
        dpt_classes_num = active_classes_num - classes_per_task
        self.FCN.train()
        '''获取imgs, examplar_imgs在pre_model的输出'''
        pre_model_output, pre_model_imgs_2_features = self.pre_tasks_model(imgs, is_nograd=True)
        '''获取imgs在要训练的模型FM'''
        outputs = self.FCN(pre_model_imgs_2_features)
        all_output = outputs["all_logits"][:, 0:active_classes_num]

        pre_model_output_for_distill = pre_model_output[:, 0:dpt_classes_num]
        all_output_for_distill = all_output[:, 0:dpt_classes_num]
        all_cls_loss = all_cls_criterion(all_output, labels)
        if self.cfg.FCTM.use_KD:
            if self.cfg.FCTM.use_binary_KD:
                distill_loss = compute_distill_binary_loss(all_output_for_distill, pre_model_output_for_distill)
            else:
                distill_loss = compute_distill_loss(all_output_for_distill, pre_model_output_for_distill,
                                                    temp=self.cfg.FCTM.T)
            loss = all_cls_loss + self.cfg.FCTM.TRAIN.tradeoff_rate * distill_loss
        else:
            loss = all_cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_task_data_result = torch.argmax(all_output, 1)
        # current_task_acc, current_task_cnt = cuda_accuracy(current_task_data_result.cpu().numpy(), labels.cpu().numpy())
        current_task_acc, current_task_cnt = cuda_accuracy(current_task_data_result, labels)
        now_acc, now_cnt = current_task_acc, current_task_cnt
        return loss, now_acc, now_cnt
        pass

    def train_main(self, train_dataset, active_classes_num, task):
        dp_classes_num = active_classes_num - self.dataset_handler.classes_per_task
        optimizer = self.build_optimize(model=self.model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.model.TRAIN.MAX_EPOCH)
        MAX_EPOCH = self.cfg.model.TRAIN.MAX_EPOCH
        best_acc = 0
        loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                            num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                            persistent_workers=True)
        for epoch in range(1, MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            is_first_ite = True
            iters_left = 1
            iter_index = 0
            iter_num = 0
            while iters_left > 0:
                # Update # iters left on current data-loader(s) and, if needed, create new one(s)
                iters_left -= 1
                if is_first_ite:
                    is_first_ite = False
                    data_loader = iter(loader)
                    # NOTE:  [train_dataset]  is training-set of current task
                    #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                    iter_num = iters_left = len(data_loader)
                    continue
                self.model.train()
                #####-----CURRENT BATCH-----#####
                try:
                    x, y = next(data_loader)  # --> sample training data of current task
                except StopIteration:
                    raise ValueError("next(data_loader) error while read data. ")
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                outputs, _ = self.model(x)
                outputs = outputs[:, 0:active_classes_num]

                pre_model_features = self.pre_tasks_model(x, is_nograd=True, feature_flag=True)
                FCTM_outputs = self.FCN(pre_model_features, is_nograd=True)
                pre_outputs = FCTM_outputs[:, 0:dp_classes_num]
                loss = compute_cls_distill_binary_loss(labels=y, output=outputs,
                                                       classes_per_task=self.dataset_handler.classes_per_task,
                                                       pre_model_output_for_distill=pre_outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || Batch:{:>3d}/{} || lr : {} || " \
                               "Loss_cls:{:>5.3f}".format(epoch, iter_index,
                                                          iter_num,
                                                          optimizer.param_groups[
                                                              0]['lr'],
                                                          all_loss.val
                                                          )
                    self.logger.info(pbar_str)
                iter_index += 1

            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:

                val_acc = self.validate_with_FC(task=task)  # task_id 从1开始
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------current_Epoch:{:>3d}    current_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()

    def first_task_train_main(self, train_dataset, active_classes_num, task_id):
        optimizer = self.build_optimize(model=self.model,
                                        base_lr=self.cfg.model.TRAIN.OPTIMIZER.BASE_LR,
                                        optimizer_type=self.cfg.model.TRAIN.OPTIMIZER.TYPE,
                                        momentum=self.cfg.model.TRAIN.OPTIMIZER.MOMENTUM,
                                        weight_decay=self.cfg.model.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        scheduler = self.build_scheduler(optimizer, lr_type=self.cfg.model.TRAIN.LR_SCHEDULER.TYPE,
                                         lr_step=self.cfg.model.TRAIN.LR_SCHEDULER.LR_STEP,
                                         lr_factor=self.cfg.model.TRAIN.LR_SCHEDULER.LR_FACTOR,
                                         warmup_epochs=self.cfg.model.TRAIN.LR_SCHEDULER.WARM_EPOCH,
                                         MAX_EPOCH=self.cfg.model.TRAIN.MAX_EPOCH)
        criterion = CrossEntropy().cuda()
        best_acc = 0
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                  num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=True, drop_last=True,
                                  persistent_workers=True)
        iter_num = len(train_loader)
        self.logger.info(f"type(self.model): {type(self.model)}.")
        for epoch in range(1, self.cfg.model.TRAIN.MAX_EPOCH + 1):
            all_loss = AverageMeter()
            if float(torch.__version__[:3]) < 1.3:
                scheduler.step()
            iter_index = 0
            for x, y in train_loader:
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)  # --> transfer them to correct device
                # ---> Train MAIN MODEL
                cnt = y.shape[0]
                output, _ = self.model(x)
                output = output[:, 0:active_classes_num]
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                all_loss.update(loss.data.item(), cnt)
                if iter_index % self.cfg.SHOW_STEP == 0:
                    pbar_str = "Epoch: {} || Batch:{:>3d}/{}|| lr: {} || Batch_cls_Loss:{:>5.3f}" \
                               "".format(epoch, iter_index, iter_num,
                                         optimizer.param_groups[0]['lr'],
                                         all_loss.val
                                         )
                    self.logger.info(pbar_str)
                iter_index += 1

            # if epoch % self.cfg.epoch_show_step == 0:
            if self.cfg.VALID_STEP != -1 and epoch % self.cfg.VALID_STEP == 0:
                pbar_str = "Validate Epoch: {} || lr: {} || epoch_Loss:{:>5.3f}".format(epoch,
                                                                                        optimizer.param_groups[0]['lr'],
                                                                                        all_loss.val)
                self.logger.info(pbar_str)

                val_acc = self.validate_with_FC(task=task_id)  # task_id 从1开始
                acc_avg = val_acc.mean()
                self.logger.info(
                    "--------------current_Epoch:{:>3d}    current_Acc:{:>5.2f}%--------------".format(
                        epoch, acc_avg * 100
                    )
                )
                if acc_avg > best_acc:
                    best_acc, best_epoch = acc_avg, epoch
                    self.best_model = copy.deepcopy(self.model)
                    self.best_epoch = best_epoch
                    self.best_acc = best_acc
                    self.logger.info(
                        "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                            best_epoch, best_acc * 100
                        )
                    )

            if float(torch.__version__[:3]) >= 1.3:
                scheduler.step()
        del train_loader

    def validate_with_exemplars(self, task, is_test=False):
        # todo
        ncm_acc = []
        centroid_acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_exemplars_per_task(self.dataset_handler.val_datasets[task_id])
            else:
                predict_result = self.validate_with_exemplars_per_task(self.dataset_handler.test_datasets[task_id])
            ncm_acc.append(predict_result[0])
            centroid_acc.append(predict_result[1])
            self.logger.info(
                f"task : {task} || per task {task_id}, ncm acc:{predict_result[0]} || centroid acc: {predict_result[1]}"
            )
        self.model.train(mode=mode)
        return np.array(ncm_acc)
        pass

    def validate_with_exemplars_per_task(self, val_dataset):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                persistent_workers=True)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        NCM_top1 = AverageMeter()
        end = time.time()

        for inputs, labels in val_loader:
            correct_temp = 0
            centroid_correct_temp = 0
            data_time.update(time.time() - end)
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            y_hat = self.exemplar_manager.classify_with_exemplars(inputs,
                                                                  self.model,
                                                                  feature_flag=True)  # x, model, classifying_approach="NCM", allowed_classes

            correct_temp += y_hat.eq(labels.data).cpu().sum()
            NCM_top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        del val_loader
        return NCM_top1.avg, 0
        pass

    def validate_with_FC(self, model=None, task=None, is_test=False):
        acc = []
        Model = model if model is not None else self.model
        mode = Model.training
        Model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task(Model, self.dataset_handler.val_datasets[task_id], task)
            else:
                predict_result = self.validate_with_FC_per_task(Model, self.dataset_handler.test_datasets[task_id],
                                                                task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        Model.train(mode=mode)
        # print(
        #     f"task {task} validate_with_exemplars, acc_avg:{acc.mean()}")
        # self.logger.info(
        #     f"per task {task}, validate_with_exemplars, avg acc:{acc.mean()}"
        #     f"-------------------------------------------------------------"
        # )
        return acc
        pass

    def validate_with_FC_per_task(self, Model, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                persistent_workers=True)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = Model(x=inputs, is_nograd=True, get_classifier=True)
            _, balance_fc_y_hat = torch.max(out[:, 0:active_classes_num], 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        del val_loader
        return top1.avg
        pass

    def validate_with_exemplars_taskIL(self, task, is_test=False):
        # todo
        ncm_acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_exemplars_per_task_taskIL(
                    self.dataset_handler.val_datasets[task_id],
                    task_id)
            else:
                predict_result = self.validate_with_exemplars_per_task_taskIL(
                    self.dataset_handler.test_datasets[task_id],
                    task_id)
            ncm_acc.append(predict_result)
            self.logger.info(
                f"task : {task} || per task {task_id}, ncm acc:{predict_result}"
            )
        self.model.train(mode=mode)
        return np.array(ncm_acc)
        pass

    def validate_with_exemplars_per_task_taskIL(self, val_dataset, task_id):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                persistent_workers=True)
        NCM_top1 = AverageMeter()
        allowed_classes = [i for i in range(task_id * self.dataset_handler.classes_per_task,
                                            (task_id + 1) * self.dataset_handler.classes_per_task)]
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            y_hat = self.exemplar_manager.classify_with_exemplars(inputs,
                                                                  self.model,
                                                                  allowed_classes=allowed_classes,
                                                                  feature_flag=True)  # x, model, classifying_approach="NCM", allowed_classes
            correct_temp += y_hat.eq(labels.data).cpu().sum()
            NCM_top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        del val_loader
        return NCM_top1.avg
        pass

    def validate_with_FC_taskIL(self, task, is_test=False):
        acc = []
        mode = self.model.training
        self.model.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.val_datasets[task_id],
                                                                       task_id, task=task)
            else:
                predict_result = self.validate_with_FC_per_task_taskIL(self.dataset_handler.test_datasets[task_id],
                                                                       task_id, task=task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.model.train(mode=mode)
        return acc
        pass

    def validate_with_FC_per_task_taskIL(self, val_dataset, task_id, task=None):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                persistent_workers=True)
        top1 = AverageMeter()
        correct = 0
        allowed_classes = [i for i in range(task_id * self.dataset_handler.classes_per_task,
                                            (task_id + 1) * self.dataset_handler.classes_per_task)]
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = self.model(x=inputs, is_nograd=True, get_classifier=True)
            _, balance_fc_y_hat = torch.max(out[:, allowed_classes], 1)
            balance_fc_y_hat += task_id * self.dataset_handler.classes_per_task
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        del val_loader
        return top1.avg
        pass

    def validate_with_FCTM(self, task, is_test=False):
        acc = []
        fcn_mode = self.FCN.training
        self.FCN.eval()
        for task_id in range(task):  # 这里的task 从0 开始
            if self.dataset_handler.val_datasets and (not is_test):
                predict_result = self.validate_with_FCTM_per_task(self.dataset_handler.val_datasets[task_id], task)
            else:
                predict_result = self.validate_with_FCTM_per_task(self.dataset_handler.test_datasets[task_id],
                                                                  task)
            acc.append(predict_result)
            self.logger.info(
                f"task: {task} || per task {task_id}, validate_with_FC acc:{predict_result}"
            )
        acc = np.array(acc)
        self.FCN.train(mode=fcn_mode)
        # print(
        #     f"task {task} validate_with_exemplars, acc_avg:{acc.mean()}")
        # self.logger.info(
        #     f"per task {task}, validate_with_exemplars, avg acc:{acc.mean()}"
        #     f"-------------------------------------------------------------"
        # )
        return acc

    def validate_with_FCTM_per_task(self, val_dataset, task):
        # todo
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.model.TRAIN.BATCH_SIZE,
                                num_workers=self.cfg.model.TRAIN.NUM_WORKERS, shuffle=False, drop_last=False,
                                persistent_workers=True)
        top1 = AverageMeter()
        correct = 0
        active_classes_num = self.dataset_handler.classes_per_task * task
        for inputs, labels in val_loader:
            correct_temp = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pre_model_features = self.pre_tasks_model(inputs, is_nograd=True, feature_flag=True)
            with torch.no_grad():
                out = self.FCN(pre_model_features, is_nograd=True)
            _, balance_fc_y_hat = torch.max(out[:, 0:active_classes_num], 1)
            correct_temp += balance_fc_y_hat.eq(labels.data).cpu().sum()
            correct += correct_temp
            top1.update((correct_temp / inputs.size(0)).item(), inputs.size(0))
        del val_loader
        return top1.avg
        pass

    def save_best_latest_model_data(self, model_dir, task_id, acc, epoch):
        if self.best_model is None:
            self.best_model = self.model
        if self.latest_model is None:
            self.latest_model = self.model
        if task_id == 1 or self.cfg.use_base_half and task_id == int(self.dataset_handler.all_tasks / 2):
            torch.save({
                'state_dict': self.best_model.state_dict(),
                'acc_result': self.best_acc,
                'best_epoch': self.best_epoch,
                'task_id': task_id
            }, os.path.join(model_dir, "base_best_model.pth")
            )
            torch.save({
                'state_dict': self.latest_model.state_dict(),
                'acc_result': acc,
                'latest_epoch': epoch,
                'task_id': task_id
            }, os.path.join(model_dir, "base_latest_model.pth")
            )
            split_selected_data = self.dataset_handler.get_split_selected_data()
            torch.save({
                'exemplar_sets': self.exemplar_manager.exemplar_sets,
                'store_original_imgs': self.exemplar_manager.store_original_imgs,
                'split_selected_data': split_selected_data
            }, os.path.join(model_dir, "base_exp_data_info.pkl")
            )
        else:
            torch.save({
                'state_dict': self.best_model.state_dict(),
                'acc_result': self.best_acc,
                'best_epoch': self.best_epoch,
                'task_id': task_id
            }, os.path.join(model_dir, "best_model.pth")
            )
            torch.save({
                'state_dict': self.latest_model.state_dict(),
                'acc_result': acc,
                'latest_epoch': epoch,
                'task_id': task_id
            }, os.path.join(model_dir, "latest_model.pth")
            )
            split_selected_data = self.dataset_handler.get_split_selected_data()
            torch.save({
                'exemplar_sets': self.exemplar_manager.exemplar_sets,
                'store_original_imgs': self.exemplar_manager.store_original_imgs,
                'split_selected_data': split_selected_data
            }, os.path.join(model_dir, "exp_data_info.pkl")
            )

        pass
