# #!/usr/bin/env python3
# import argparse
# import os
# import numpy as np
# import time
# import torch
#
# import visual_plt
# from public import utils
# import pandas as pd
# from CIFAR.alg_model import EFAfIL
# from param_stamp import get_param_stamp_from_args
# import evaluate
# from public.data import get_multitask_experiment
# from param_values import set_default_values
#
# parser = argparse.ArgumentParser('./main.py', description='Run individual continual learning experiment.')
# parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
# parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
# parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
# parser.add_argument('--availabel_cudas', help="don't use GPUs")
# parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help="default: %(default)s")
# parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
# parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")
#
# # expirimental task parameters
# task_params = parser.add_argument_group('Task Parameters')
# task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST'])
# task_params.add_argument('--scenario', type=str, default='class', choices=['task', 'domain', 'class'])
# task_params.add_argument('--tasks', type=int, help='number of tasks')
#
# # specify loss functions to be used
# loss_params = parser.add_argument_group('Loss Parameters')
#
# loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")
# loss_params.add_argument('--bce-distill', action='store_true', help='distilled loss on previous classes for new'                                                                 ' examples (only if --bce & --scenario="class")')
# loss_params.add_argument('--kd_temp', type=float, help="distill temp")
#
# # model architecture parameters
# model_params = parser.add_argument_group('MLP Parameters')
# model_params.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
# model_params.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")
# model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
# model_params.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
# model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
#
# # training hyperparameters / initialization
# train_params = parser.add_argument_group('Training Parameters')
# train_params.add_argument('--epochs', type=int, help="# epochs to optimize solver")
# train_params.add_argument('--lr', type=float, help="learning rate")
# train_params.add_argument('--batch', type=int, default=128, help="batch-size")
# train_params.add_argument('--momentum', type=float, default=0.9, help="momentum")
# train_params.add_argument('--weight_decay', type=float, default=1e-5, help="momentum")
# train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')
#
# # "memory replay" parameters
# replay_params = parser.add_argument_group('Replay Parameters')
# replay_params.add_argument('--distill', action='store_true', help="use distillation for replay?")
# replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
#
# # data storage ('exemplars') parameters
# store_params = parser.add_argument_group('Data Storage Parameters')
# store_params.add_argument('--icarl', action='store_true', help="bce-distill, use-exemplars & add-exemplars")
# store_params.add_argument('--use-exemplars', action='store_true', help="use exemplars for classification")
# store_params.add_argument('--add-exemplars', action='store_true', help="add exemplars to current task's training set")
# store_params.add_argument('--budget', type=int, default=1000, dest="budget", help="how many samples can be stored?")
# store_params.add_argument('--herding', action='store_true',
#                           help="use herding to select stored data (instead of random)")
# store_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")
#
# # evaluation parameters
# eval_params = parser.add_argument_group('Evaluation Parameters')
# eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
# eval_params.add_argument('--metrics', action='store_true', help="calculate additional metrics (e.g., BWT, forgetting)")
# eval_params.add_argument('--pdf', action='store_true', help="generate pdf with results")
# eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
# eval_params.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
# eval_params.add_argument('--loss-log', type=int, default=200, metavar="N", help="# iters after which to plot loss")
# eval_params.add_argument('--prec-log', type=int, default=200, metavar="N", help="# iters after which to plot precision")
# eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
# eval_params.add_argument('--sample-log', type=int, default=500, metavar="N", help="# iters after which to plot samples")
# eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")
#
#
# def run(args, verbose=False):
#     # -if [log_per_task], reset all logs
#     if args.log_per_task:
#         args.prec_log = args.iters
#         args.loss_log = args.iters
#         args.sample_log = args.iters
#     # -if [iCaRL] is selected, select all accompanying options
#     if hasattr(args, "icarl") and args.icarl:
#         args.use_exemplars = True
#         args.add_exemplars = True
#         args.bce = True
#         args.bce_distill = True
#     # -if 'BCEdistill' is selected for other than scenario=="class", give error
#     if args.bce_distill and not args.scenario == "class":
#         raise ValueError("BCE-distill can only be used for class-incremental learning.")
#     # -create plots- and results-directories if needed
#     if not os.path.isdir(args.r_dir):
#         os.mkdir(args.r_dir)
#     if args.pdf and not os.path.isdir(args.p_dir):
#         os.mkdir(args.p_dir)
#
#     scenario = args.scenario
#
#     # If only want param-stamp, get it printed to screen and exit
#     if hasattr(args, "get_stamp") and args.get_stamp:
#         print(get_param_stamp_from_args(args=args))
#         exit()
#
#     # Use cuda?
#     cuda = torch.cuda.is_available() and args.cuda
#     device = torch.device("cuda" if cuda else "cpu")
#     if verbose:
#         print("CUDA is {}used".format("" if cuda else "NOT(!!) "))
#
#     # Set random seeds
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if cuda:
#         torch.cuda.manual_seed(args.seed)
#
#     # -------------------------------------------------------------------------------------------------#
#
#     # ----------------#
#     # ----- DATA -----#
#     # ----------------#
#
#     # Prepare data for chosen experiment
#     if verbose:
#         print("\nPreparing the data...")
#     (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
#         name=args.experiment, scenario=scenario, tasks=args.tasks, data_dir=args.d_dir,
#         verbose=verbose, exception=True if args.seed == 0 else False,
#     )
#
#     # -------------------------------------------------------------------------------------------------#
#
#     # ------------------------------#
#     # ----- MODEL (CLASSIFIER) -----#
#     # ------------------------------#
#
#     # Define main model (i.e., classifier, if requested with feedback connections)
#     model = None
#     extracted_layers = ["layer4", "avgpool", "fc"]
#     if "CIFAR100" in args.experiment:
#         if args.icarl:
#             model = EFAfIL.iCaRL(100, EFAfIL.resnet34(100, 1 / 4), extracted_layers, args.memory_budget,
#                                  norm_exemplars=args.norm_exemplars, herding=args.herding, lr=args.lr,
#                                  momentum=args.momentum, weight_decay=args.weight_decay, optim_type=args.optimizer,
#                                  KD_temp=args.kd_temp, availabel_cudas=args.availabel_cudas)
#             model.train_main(train_datasets, test_datasets, classes_per_task=classes_per_task, epochs=args.epochs,
#                     batch_size=args.size, use_exemplars=True, add_exemplars=True)
#         elif args.ILtFA:
#             # todo ILtFA
#             model = EFAfIL.ILtFA(100, EFAfIL.resnet34(100, 1 / 4), extracted_layers, args.memory_budget,
#                                  norm_exemplars=args.norm_exemplars, herding=args.herding, lr=args.lr,
#                                  momentum=args.momentum, weight_decay=args.weight_decay, optim_type=args.optimizer,
#                                  KD_temp=args.kd_temp, availabel_cudas=args.availabel_cudas)
#             model.train_main(train_datasets, test_datasets, classes_per_task=classes_per_task, epochs=args.epochs,
#                              batch_size=args.size, use_exemplars=True, add_exemplars=True)
#             pass
#         elif args.EFAfIL:
#             # todo EFAfIL
#             model = EFAfIL.EFAfIL(100, EFAfIL.resnet34(100, 1 / 4), extracted_layers, args.memory_budget,
#                                   norm_exemplars=args.norm_exemplars, herding=args.herding, lr=args.lr,
#                                   momentum=args.momentum, weight_decay=args.weight_decay, optim_type=args.optimizer,
#                                   KD_temp=args.kd_temp, availabel_cudas=args.availabel_cudas)
#             model.train_main(train_datasets, test_datasets, classes_per_task=classes_per_task, epochs=args.epochs,
#                              batch_size=args.size, use_exemplars=True, add_exemplars=True)
#             pass
#     elif "ImageNet1000" in args.experiment:
#         if args.icarl:
#             model = EFAfIL.iCaRL(1000, EFAfIL.resnet18(1000, 1), extracted_layers, args.memory_budget,
#                                  norm_exemplars=args.norm_exemplars, herding=args.herding, lr=args.lr,
#                                  momentum=args.momentum, weight_decay=args.weight_decay, optim_type=args.optimizer,
#                                  KD_temp=args.kd_temp, availabel_cudas=args.availabel_cudas)
#             model.train_main(train_datasets, test_datasets, classes_per_task=classes_per_task, epochs=args.epochs,
#                              batch_size=args.size, use_exemplars=True, add_exemplars=True)
#         elif args.ILtFA:
#             # todo ILtFA
#             model = EFAfIL.ILtFA(100, EFAfIL.resnet34(100, 1), extracted_layers, args.memory_budget,
#                                  norm_exemplars=args.norm_exemplars, herding=args.herding, lr=args.lr,
#                                  momentum=args.momentum, weight_decay=args.weight_decay, optim_type=args.optimizer,
#                                  KD_temp=args.kd_temp, availabel_cudas=args.availabel_cudas)
#             model.train_main(train_datasets, test_datasets, classes_per_task=classes_per_task, epochs=args.epochs,
#                              batch_size=args.size, use_exemplars=True, add_exemplars=True)
#             pass
#         elif args.EFAfIL:
#             # todo EFAfIL
#             pass
#     elif "ImageNet100" in args.experiment:
#         if args.icarl:
#             model = EFAfIL.iCaRL(100, EFAfIL.resnet18(100, 1), extracted_layers, args.memory_budget,
#                                  norm_exemplars=args.norm_exemplars, herding=args.herding, lr=args.lr,
#                                  momentum=args.momentum, weight_decay=args.weight_decay, optim_type=args.optimizer,
#                                  KD_temp=args.kd_temp, availabel_cudas=args.availabel_cudas)
#             model.train_main(train_datasets, test_datasets, classes_per_task=classes_per_task, epochs=args.epochs,
#                              batch_size=args.size, use_exemplars=True, add_exemplars=True)
#         elif args.ILtFA:
#             # todo ILtFA
#             pass
#         elif args.EFAfIL:
#             # todo EFAfIL
#             pass
#
#     # -------------------------------------------------------------------------------------------------#
#
#     # --------------------#
#     # ----- TRAINING -----#
#     # --------------------#
#
#     if verbose:
#         print("\nTraining...")
#     # Keep track of training-time
#     start = time.time()
#     # Train model
#     model.train_cl(
#         model, train_datasets, replay_mode=args.replay, scenario=scenario, classes_per_task=classes_per_task,
#         iters=args.iters, batch_size=args.batch,
#         generator=generator, gen_iters=args.g_iters, gen_loss_cbs=generator_loss_cbs,
#         sample_cbs=sample_cbs, eval_cbs=eval_cbs, loss_cbs=generator_loss_cbs if args.feedback else solver_loss_cbs,
#         metric_cbs=metric_cbs, use_exemplars=args.use_exemplars, add_exemplars=args.add_exemplars,
#     )
#     # Get total training-time in seconds, and write to file
#     if args.time:
#         training_time = time.time() - start
#         time_file = open("{}/time-{}.txt".format(args.r_dir, param_stamp), 'w')
#         time_file.write('{}\n'.format(training_time))
#         time_file.close()
#
#     # -------------------------------------------------------------------------------------------------#
#
#     # ----------------------#
#     # ----- EVALUATION -----#
#     # ----------------------#
#
#     if verbose:
#         print("\n\nEVALUATION RESULTS:")
#
#     # Evaluate precision of final model on full test-set
#     precs = [evaluate.validate(
#         model, test_datasets[i], verbose=False, test_size=None, task=i + 1, with_exemplars=False,
#         allowed_classes=list(range(classes_per_task * i, classes_per_task * (i + 1))) if scenario == "task" else None
#     ) for i in range(args.tasks)]
#     average_precs = sum(precs) / args.tasks
#     # -print on screen
#     if verbose:
#         print("\n Precision on test-set{}:".format(" (softmax classification)" if args.use_exemplars else ""))
#         for i in range(args.tasks):
#             print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
#         print('=> Average precision over all {} tasks: {:.4f}\n'.format(args.tasks, average_precs))
#
#     # -with exemplars
#     if args.use_exemplars:
#         precs = [evaluate.validate(
#             model, test_datasets[i], verbose=False, test_size=None, task=i + 1, with_exemplars=True,
#             allowed_classes=list(
#                 range(classes_per_task * i, classes_per_task * (i + 1))) if scenario == "task" else None
#         ) for i in range(args.tasks)]
#         average_precs_ex = sum(precs) / args.tasks
#         # -print on screen
#         if verbose:
#             print(" Precision on test-set (classification using exemplars):")
#             for i in range(args.tasks):
#                 print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
#             print('=> Average precision over all {} tasks: {:.4f}\n'.format(args.tasks, average_precs_ex))
#
#     if args.metrics:
#         # Accuracy matrix
#         if args.scenario in ('task', 'domain'):
#             R = pd.DataFrame(data=metrics_dict['acc per task'],
#                              index=['after task {}'.format(i + 1) for i in range(args.tasks)])
#             R.loc['at start'] = metrics_dict['initial acc per task'] if (not args.use_exemplars) else [
#                 'NA' for _ in range(args.tasks)
#             ]
#             R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
#             BWTs = [(R.loc['after task {}'.format(args.tasks), 'task {}'.format(i + 1)] - \
#                      R.loc['after task {}'.format(i + 1), 'task {}'.format(i + 1)]) for i in range(args.tasks - 1)]
#             FWTs = [0. if args.use_exemplars else (
#                     R.loc['after task {}'.format(i + 1), 'task {}'.format(i + 2)] - R.loc[
#                 'at start', 'task {}'.format(i + 2)]
#             ) for i in range(args.tasks - 1)]
#             forgetting = []
#             for i in range(args.tasks - 1):
#                 forgetting.append(max(R.iloc[1:args.tasks, i]) - R.iloc[args.tasks, i])
#             R.loc['FWT (per task)'] = ['NA'] + FWTs
#             R.loc['BWT (per task)'] = BWTs + ['NA']
#             R.loc['F (per task)'] = forgetting + ['NA']
#             BWT = sum(BWTs) / (args.tasks - 1)
#             F = sum(forgetting) / (args.tasks - 1)
#             FWT = sum(FWTs) / (args.tasks - 1)
#             metrics_dict['BWT'] = BWT
#             metrics_dict['F'] = F
#             metrics_dict['FWT'] = FWT
#             # -print on screen
#             if verbose:
#                 print("Accuracy matrix")
#                 print(R)
#                 print("\nFWT = {:.4f}".format(FWT))
#                 print("BWT = {:.4f}".format(BWT))
#                 print("  F = {:.4f}\n\n".format(F))
#         else:
#             if verbose:
#                 # Accuracy matrix based only on classes in that task (i.e., evaluation as if Task-IL scenario)
#                 R = pd.DataFrame(data=metrics_dict['acc per task (only classes in task)'],
#                                  index=['after task {}'.format(i + 1) for i in range(args.tasks)])
#                 R.loc['at start'] = metrics_dict[
#                     'initial acc per task (only classes in task)'
#                 ] if not args.use_exemplars else ['NA' for _ in range(args.tasks)]
#                 R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
#                 print("Accuracy matrix, based on only classes in that task ('as if Task-IL scenario')")
#                 print(R)
#
#                 # Accuracy matrix, always based on all classes
#                 R = pd.DataFrame(data=metrics_dict['acc per task (all classes)'],
#                                  index=['after task {}'.format(i + 1) for i in range(args.tasks)])
#                 R.loc['at start'] = metrics_dict[
#                     'initial acc per task (only classes in task)'
#                 ] if not args.use_exemplars else ['NA' for _ in range(args.tasks)]
#                 R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
#                 print("\nAccuracy matrix, always based on all classes")
#                 print(R)
#
#                 # Accuracy matrix, based on all classes thus far
#                 R = pd.DataFrame(data=metrics_dict['acc per task (all classes up to trained task)'],
#                                  index=['after task {}'.format(i + 1) for i in range(args.tasks)])
#                 print("\nAccuracy matrix, based on all classes up to the trained task")
#                 print(R)
#
#             # Accuracy matrix, based on all classes up to the task being evaluated
#             # (this is the accuracy-matrix used for calculating the metrics in the Class-IL scenario)
#             R = pd.DataFrame(data=metrics_dict['acc per task (all classes up to evaluated task)'],
#                              index=['after task {}'.format(i + 1) for i in range(args.tasks)])
#             R.loc['at start'] = metrics_dict[
#                 'initial acc per task (only classes in task)'
#             ] if not args.use_exemplars else ['NA' for _ in range(args.tasks)]
#             R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
#             BWTs = [(R.loc['after task {}'.format(args.tasks), 'task {}'.format(i + 1)] - \
#                      R.loc['after task {}'.format(i + 1), 'task {}'.format(i + 1)]) for i in range(args.tasks - 1)]
#             FWTs = [0. if args.use_exemplars else (
#                     R.loc['after task {}'.format(i + 1), 'task {}'.format(i + 2)] - R.loc[
#                 'at start', 'task {}'.format(i + 2)]
#             ) for i in range(args.tasks - 1)]
#             forgetting = []
#             for i in range(args.tasks - 1):
#                 forgetting.append(max(R.iloc[1:args.tasks, i]) - R.iloc[args.tasks, i])
#             R.loc['FWT (per task)'] = ['NA'] + FWTs
#             R.loc['BWT (per task)'] = BWTs + ['NA']
#             R.loc['F (per task)'] = forgetting + ['NA']
#             BWT = sum(BWTs) / (args.tasks - 1)
#             F = sum(forgetting) / (args.tasks - 1)
#             FWT = sum(FWTs) / (args.tasks - 1)
#             metrics_dict['BWT'] = BWT
#             metrics_dict['F'] = F
#             metrics_dict['FWT'] = FWT
#             # -print on screen
#             if verbose:
#                 print("\nAccuracy matrix, based on all classes up to the evaluated task")
#                 print(R)
#                 print("\n=> FWT = {:.4f}".format(FWT))
#                 print("=> BWT = {:.4f}".format(BWT))
#                 print("=>  F = {:.4f}\n".format(F))
#
#     if verbose and args.time:
#         print("=> Total training time = {:.1f} seconds\n".format(training_time))
#
#     # -------------------------------------------------------------------------------------------------#
#
#     # ------------------#
#     # ----- OUTPUT -----#
#     # ------------------#
#
#     # Average precision on full test set
#     output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'w')
#     output_file.write('{}\n'.format(average_precs_ex if args.use_exemplars else average_precs))
#     output_file.close()
#     # -metrics-dict
#     if args.metrics:
#         file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
#         utils.save_object(metrics_dict, file_name)
#
#     # -------------------------------------------------------------------------------------------------#
#
#     # --------------------#
#     # ----- PLOTTING -----#
#     # --------------------#
#
#     # If requested, generate pdf
#     if args.pdf:
#         # -open pdf
#         plot_name = "{}/{}.pdf".format(args.p_dir, param_stamp)
#         pp = visual_plt.open_pdf(plot_name)
#
#         # -show samples and reconstructions (either from main model or from separate generator)
#         if args.feedback or args.replay == "generative":
#             evaluate.show_samples(model if args.feedback else generator, config, size=args.sample_n, pdf=pp)
#             for i in range(args.tasks):
#                 evaluate.show_reconstruction(model if args.feedback else generator, test_datasets[i], config, pdf=pp,
#                                              task=i + 1)
#
#         # -show metrics reflecting progression during training
#         figure_list = []  # -> create list to store all figures to be plotted
#
#         # -generate all figures (and store them in [figure_list])
#         key = "acc per task ({} task)".format("all classes up to trained" if scenario == 'class' else "only classes in")
#         plot_list = []
#         for i in range(args.tasks):
#             plot_list.append(metrics_dict[key]["task {}".format(i + 1)])
#         figure = visual_plt.plot_lines(
#             plot_list, x_axes=metrics_dict["x_task"],
#             line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
#         )
#         figure_list.append(figure)
#         figure = visual_plt.plot_lines(
#             [metrics_dict["average"]], x_axes=metrics_dict["x_task"],
#             line_names=['average all tasks so far']
#         )
#         figure_list.append(figure)
#
#         # -add figures to pdf (and close this pdf).
#         for figure in figure_list:
#             pp.savefig(figure)
#
#         # -close pdf
#         pp.close()
#
#         # -print name of generated plot on screen
#         if verbose:
#             print("\nGenerated plot: {}\n".format(plot_name))
#
#
# if __name__ == '__main__':
#     # -load input-arguments
#     args = parser.parse_args()
#     # -set default-values for certain arguments based on chosen scenario & experiment
#     args = set_default_values(args)
#     # -run experiment
#     run(args, verbose=True)
