import argparse
import copy
import time
import os
import datetime
import warnings

from model import *
from utils import *
from baselines import run_mlr_baseline, run_xgboost_baseline

warnings.filterwarnings("ignore")  # silence all warnings


def init_seed(seed=None):
    """
    Initialize random seeds for NumPy and PyTorch to make runs reproducible.

    If seed is None, we generate one from the current time.
    """
    if seed is None:
        seed = int(time.time() * 1000 // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Argument parser / Hyperparameters
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='VED', help='VED/ETTD')

# Which model to run
parser.add_argument(
    '--baseline_type',
    type=str,
    default='meta_pec',
    choices=['meta_pec', 'mlr', 'xgboost'],
    help='Choose which model to run.'
)

# General embedding & feature dimensions
parser.add_argument('--dim', type=int, default=20)
parser.add_argument('--car_fea_dim', type=int, default=7)
parser.add_argument('--feature_emb_dim', type=int, default=10)
parser.add_argument('--car_dim', type=int, default=10)
parser.add_argument('--car_type_dim', type=int, default=20)
parser.add_argument('--traj_fea_dim', type=int, default=7)

# Number of entities (cars, types, roads, etc.)
parser.add_argument('--car_num', type=int, default=358)
parser.add_argument('--car_type_num', type=int, default=5)
parser.add_argument('--road_dim', type=int, default=10)
parser.add_argument('--road_fea_dim', type=int, default=5)
parser.add_argument('--road_fea_dim_ori', type=int, default=14)
parser.add_argument('--road_emb_dim', type=int, default=10)
parser.add_argument('--road_type_dim', type=int, default=10)
parser.add_argument('--road_type_num', type=int, default=15)

# Driving / trajectory related dimensions
parser.add_argument('--drive_fea_dim', type=int, default=3)
parser.add_argument('--max_tra_num', type=int, default=5)
parser.add_argument('--max_traj_len', type=int, default=160)
parser.add_argument('--max_task_data_num', type=int, default=50)

# Discretization limits for time and location
parser.add_argument('--max_time', type=int, default=8000)
parser.add_argument('--max_location', type=int, default=15)

# Model architecture hyperparameters
parser.add_argument('--filter_size', type=int, default=2)
parser.add_argument('--head_num', type=int, default=4)

# Training setup
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--base_epoch', type=int, default=1)
parser.add_argument('--activation', type=str, default='leakyrelu')
parser.add_argument('--GRU_drop', type=float, default=0.0)
parser.add_argument('--traj_drop', type=float, default=0.1)
parser.add_argument('--encoder_drop', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--task_batch_num', type=int, default=18)

# Learning rates
parser.add_argument('--meta_lr', type=float, default=0.006, help='learning rate for meta-learner.')
parser.add_argument('--lr', type=float, default=0.0006, help='inner learning rate.')
parser.add_argument('--fine_tune_lr', type=float, default=0.0006, help='learning rate for per-user fine-tuning.')

# Regularization and activations
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

# Early stopping parameters
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--fine_tune_patience', type=int, default=10)

# Where to save the best model
parser.add_argument('--model_path', type=str, default='./best_model',
                    help='directory to store best model parameters.')

opt = parser.parse_args()

# Make sure save dir exists
os.makedirs(opt.model_path, exist_ok=True)

# =========================================================
# Dataset-specific overrides for ETTD
# =========================================================
if opt.dataset == "ETTD":
    opt.car_num = 381
    opt.car_type_num = 1
    opt.road_type_num = 80
    opt.car_fea_dim = 6
    opt.traj_fea_dim = 6
    opt.max_tra_num = 5
    opt.max_traj_len = 16
    opt.road_fea_dim_ori = 8
    opt.task_batch_num = 36
    opt.filter_size = 4
    opt.meta_lr = 0.003
    opt.lr = 0.0001
    opt.fine_tune_lr = 0.0
    opt.patience = 5
    opt.epoch = 10
    opt.max_task_data_num = 30
    opt.max_time = 15000
    opt.max_location = 25


def meta_learning(maml, valid_data):
    """
    Run the meta-learning (MAML-style) training loop.
    """
    print('-------------------------------------------------------')
    print("Meta-Learning...")
    start = time.time()

    best_result = [99999, 99999, 99999]  # [best_MSE, best_MAE, best_MAPE]
    best_epoch = [0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        print('start training: ', datetime.datetime.now())

        maml.train_tasks()

        MSE, MAE, MAPE = test(maml.maml, opt, valid_data, isTestset=False)

        flag = 0
        if MSE <= best_result[0]:
            best_result[0] = MSE
            best_epoch[0] = epoch
            flag = 1
        if MAE <= best_result[1]:
            best_result[1] = MAE
            best_epoch[1] = epoch
            flag = 1
        if MAPE <= best_result[2]:
            best_result[2] = MAPE
            best_epoch[2] = epoch
            flag = 1
            torch.save(maml.model.state_dict(), f"{opt.model_path}/{opt.dataset}.pth")

        print('Current Result:')
        print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' % (MSE, MAE, MAPE))
        print('Best Result:')
        print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f\tEpoch:\t%d, %d, %d' %
              (best_result[0], best_result[1], best_result[2],
               best_epoch[0], best_epoch[1], best_epoch[2]))

        if flag != 1:
            bad_counter += 1
            if bad_counter >= opt.patience:
                break
        else:
            bad_counter = 0

    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


def fine_tuning_user(train_data, valid_data, test_data):
    """
    Fine-tune the meta-learned model for a single user.
    """
    model = Pec(opt)
    model.load_state_dict(torch.load(f"{opt.model_path}/{opt.dataset}.pth"))
    model = trans_to_cuda(model)

    start = time.time()

    best_result = [99999, 99999, 99999]
    best_epoch = [0, 0, 0]
    bad_counter = 0
    test_score = [0, 0, 0]

    if opt.dataset == 'ETTD':
        t_MSE, t_MAE, t_MAPE = test(model, opt, test_data)
        test_score[0], test_score[1], test_score[2] = t_MSE, t_MAE, t_MAPE
    else:
        for epoch in range(opt.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)

            MSE, MAE, MAPE = train_test(model, opt, train_data, valid_data)

            flag_mape = 0
            flag_patience = 0

            if MSE <= best_result[0]:
                best_result[0] = MSE
                best_epoch[0] = epoch
                flag_patience = 1
            if MAE <= best_result[1]:
                best_result[1] = MAE
                best_epoch[1] = epoch
                flag_patience = 1
            if MAPE <= best_result[2]:
                best_result[2] = MAPE
                best_epoch[2] = epoch
                flag_mape = 1
                flag_patience = 1

            print('Current Result:')
            print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' % (MSE, MAE, MAPE))
            print('Best Result:')
            print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f\tEpoch:\t%d, %d, %d' %
                  (best_result[0], best_result[1], best_result[2],
                   best_epoch[0], best_epoch[1], best_epoch[2]))

            if flag_patience != 1:
                bad_counter += 1
                if bad_counter >= opt.fine_tune_patience:
                    break
            else:
                if flag_mape == 1:
                    t_MSE, t_MAE, t_MAPE = test(model, opt, test_data)
                    test_score[0], test_score[1], test_score[2] = t_MSE, t_MAE, t_MAPE
                    print('Test Result:')
                    print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' %
                          (test_score[0], test_score[1], test_score[2]))

    print('-------------------------------------------------------')
    print('Test Result:')
    print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' %
          (test_score[0], test_score[1], test_score[2]))
    end = time.time()
    print("Run time: %f s" % (end - start))

    return test_score


def run_meta_pec():
    """
    Original Meta-Pec pipeline:
    1) Set random seed.
    2) Load meta-training and validation data + trajectories.
    3) Wrap raw JSON entries into Data objects.
    4) Initialize MetaLearner and run meta-learning.
    5) Load per-user fine-tuning and test data.
    6) Fine-tune separately for each user.
    7) Print overall average performance across all users.
    """
    # ------------------ Meta-training data ------------------
    train_data = load_json(f"./datasets/{opt.dataset}/train_meta.json")
    valid_data = load_json(f"./datasets/{opt.dataset}/valid_meta.json")
    trajectories = load_json(f"./datasets/{opt.dataset}/trajectories.json")

    # Convert each split (support/query) into a Data object
    for i in range(len(train_data)):
        train_data[i][0] = Data(train_data[i][0], trajectories, opt)
        train_data[i][1] = Data(train_data[i][1], trajectories, opt)

    valid_data = Data(valid_data, trajectories, opt)

    print(opt)

    # Step 1: Meta-learning
    maml = MetaLearner(train_data, opt)
    meta_learning(maml, valid_data)

    # Step 2: Per-user fine-tuning
    train_data_fine_tune = load_json(f"./datasets/{opt.dataset}/train_fine_tune.json")
    test_data = load_json(f"./datasets/{opt.dataset}/test_meta.json")

    for u in train_data_fine_tune:
        train_data_fine_tune[u][0] = Data(train_data_fine_tune[u][0], trajectories, opt)
        train_data_fine_tune[u][1] = Data(train_data_fine_tune[u][1], trajectories, opt)

    for u in test_data:
        test_data[u] = Data(test_data[u], trajectories, opt)

    all_user_scores = [[], [], []]

    for u in test_data:
        print('-------------------------------------------------------')
        print(f"Fine tuning on user {u}...")
        train_data_u, valid_data_u = train_data_fine_tune[u]
        user_test_data = test_data[u]

        user_test_score = fine_tuning_user(train_data_u, valid_data_u, user_test_data)

        for i in range(len(all_user_scores)):
            all_user_scores[i].append(user_test_score[i])

    print('-------------------------------------------------------')
    print('All User Test Result:')
    print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' %
          (np.mean(all_user_scores[0]),
           np.mean(all_user_scores[1]),
           np.mean(all_user_scores[2])))

    return (
        np.mean(all_user_scores[0]),
        np.mean(all_user_scores[1]),
        np.mean(all_user_scores[2])
    )


def main():
    init_seed(2023)

    if opt.baseline_type == 'mlr':
        print(opt)
        return run_mlr_baseline(opt)

    if opt.baseline_type == 'xgboost':
        print(opt)
        return run_xgboost_baseline(opt)

    return run_meta_pec()


if __name__ == '__main__':
    main()