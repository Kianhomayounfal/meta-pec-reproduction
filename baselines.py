import numpy as np
from collections import defaultdict

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils import load_json


def compute_metrics(y_true, y_pred, eps=1e-8):
    """
    Compute MSE, MAE, and MAPE in the same style as the main project.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0
    return mse, mae, mape


def summarize_array(arr):
    """
    Return a compact statistical summary of a 1D numeric array.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
    ]


def summarize_matrix(mat):
    """
    Return per-column mean summary flattened into one vector.
    This keeps feature size fixed even when trip length varies.
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.size == 0:
        return []

    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    return mat.mean(axis=0).astype(np.float64).tolist()


def extract_baseline_features(sample, dataset):
    """
    Convert one raw sample from the JSON files into one fixed-size feature vector.

    Raw sample format from your project:
    [cars, car_types, features, road_features, road_types, road_idx, fuel, traj_idxs]

    We use:
    - trip/car-level features: 'features'
    - car type
    - route length
    - summary stats of road features
    - summary stats of driving-related road features
    - summary stats of road types
    """
    cars, car_types, features, road_features, road_types, road_idx, fuel, traj_idxs = sample

    x = []

    # ---------------------------
    # Global / trip-level features
    # ---------------------------
    x.append(float(car_types))  # include car type, but not raw car id
    x.extend(np.asarray(features, dtype=np.float64).tolist())

    # Number of roads in route
    x.append(float(len(road_types)))
    x.append(float(len(road_idx)) if road_idx is not None else 0.0)
    x.append(float(len(traj_idxs)) if traj_idxs is not None else 0.0)

    # ---------------------------
    # Road features
    # ---------------------------
    road_features = np.asarray(road_features, dtype=np.float64)
    road_types = np.asarray(road_types, dtype=np.float64)

    if road_features.ndim == 1:
        road_features = road_features.reshape(1, -1)

    # In your Data class:
    # VED  -> first 5 cols are road features, cols 11: are drive features
    # ETTD -> first 5 cols are road features, cols 5: are drive features
    base_road_features = road_features[:, :5] if road_features.shape[1] >= 5 else road_features

    if dataset == "VED":
        drive_features = road_features[:, 11:] if road_features.shape[1] > 11 else np.empty((road_features.shape[0], 0))
    else:
        drive_features = road_features[:, 5:] if road_features.shape[1] > 5 else np.empty((road_features.shape[0], 0))

    # Aggregate road features columnwise
    x.extend(summarize_matrix(base_road_features))

    # Aggregate drive features columnwise
    if drive_features.size > 0:
        x.extend(summarize_matrix(drive_features))
    else:
        # keep dimensional consistency if empty
        x.extend([])

    # Aggregate road type statistics
    x.extend(summarize_array(road_types))

    # ---------------------------
    # Target
    # ---------------------------
    y = float(fuel)

    return x, y


def build_tabular_train_test(opt):
    """
    Build train/test arrays using the exact same split files that Meta-Pec uses:
    - train_fine_tune.json
    - test_meta.json

    Training data:
      all users' train + valid splits from train_fine_tune.json

    Test data:
      all users' test samples from test_meta.json

    We also keep test_users so we can report mean per-user metrics.
    """
    train_data_fine_tune = load_json(f"./datasets/{opt.dataset}/train_fine_tune.json")
    test_data = load_json(f"./datasets/{opt.dataset}/test_meta.json")

    X_train, y_train = [], []
    X_test, y_test = [], []
    test_users = []

    # Build training set
    for u in train_data_fine_tune:
        user_train = train_data_fine_tune[u][0]
        user_valid = train_data_fine_tune[u][1]

        # each split is the raw tuple-format dataset:
        # [cars, car_types, features, road_features, road_types, road_idx, fuel, traj_idxs]
        train_samples = list(zip(*user_train))
        valid_samples = list(zip(*user_valid))

        for sample in train_samples + valid_samples:
            x, y = extract_baseline_features(sample, opt.dataset)
            X_train.append(x)
            y_train.append(y)

    # Build test set
    for u in test_data:
        user_test = test_data[u]
        test_samples = list(zip(*user_test))

        for sample in test_samples:
            x, y = extract_baseline_features(sample, opt.dataset)
            X_test.append(x)
            y_test.append(y)
            test_users.append(u)

    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.float64),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.float64),
        test_users
    )


def evaluate_predictions_per_user(y_true, y_pred, test_users):
    """
    Compute average per-user MSE, MAE, MAPE to match the spirit of your Meta-Pec evaluation.
    """
    user_true = defaultdict(list)
    user_pred = defaultdict(list)

    for u, yt, yp in zip(test_users, y_true, y_pred):
        user_true[u].append(float(yt))
        user_pred[u].append(float(yp))

    per_user_scores = [[], [], []]  # mse, mae, mape

    for u in user_true:
        mse, mae, mape = compute_metrics(user_true[u], user_pred[u])
        per_user_scores[0].append(mse)
        per_user_scores[1].append(mae)
        per_user_scores[2].append(mape)

    return (
        float(np.mean(per_user_scores[0])),
        float(np.mean(per_user_scores[1])),
        float(np.mean(per_user_scores[2]))
    )


def run_mlr_baseline(opt):
    """
    Run Multiple Linear Regression baseline.
    Standardization helps LR behave more stably.
    """
    print('-------------------------------------------------------')
    print("Running MLR baseline...")

    X_train, y_train, X_test, y_test, test_users = build_tabular_train_test(opt)

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape: ", X_test.shape, y_test.shape)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    overall_mse, overall_mae, overall_mape = compute_metrics(y_test, y_pred)
    user_mse, user_mae, user_mape = evaluate_predictions_per_user(y_test, y_pred, test_users)

    print('-------------------------------------------------------')
    print('MLR Overall Test Result:')
    print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' %
          (overall_mse, overall_mae, overall_mape))

    print('MLR Average Per-User Test Result:')
    print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' %
          (user_mse, user_mae, user_mape))

    return user_mse, user_mae, user_mape


def run_xgboost_baseline(opt):
    """
    Run XGBoost regression baseline.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Please run: pip install xgboost")

    print('-------------------------------------------------------')
    print("Running XGBoost baseline...")

    X_train, y_train, X_test, y_test, test_users = build_tabular_train_test(opt)

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape: ", X_test.shape, y_test.shape)

    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=2023,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    overall_mse, overall_mae, overall_mape = compute_metrics(y_test, y_pred)
    user_mse, user_mae, user_mape = evaluate_predictions_per_user(y_test, y_pred, test_users)

    print('-------------------------------------------------------')
    print('XGBoost Overall Test Result:')
    print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' %
          (overall_mse, overall_mae, overall_mape))

    print('XGBoost Average Per-User Test Result:')
    print('\tMSE:\t%.4f\tMAE:\t%.4f\tMAPE:\t%.4f' %
          (user_mse, user_mae, user_mape))

    return user_mse, user_mae, user_mape