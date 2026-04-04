import os
import sys
from typing import Any

import numpy as np
import torch

sys.path.append("util")

import argument
from myNN import myNN

try:
    from prettyPlot.parser import parse_input_file  # type: ignore
except Exception:
    def parse_input_file(filepath: str) -> dict[str, str]:
        """Fallback parser for the original colon-separated input file format."""
        out: dict[str, str] = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("!") or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                out[key.strip()] = value.strip()
        return out


def _normalize_optional_path(value: Any):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.upper() in {"NONE", "NULL", ""}:
            return None
        return stripped
    return value


def absolute_path_check(path):
    if path is None:
        absolute = True
    elif os.path.isabs(path):
        absolute = True
    else:
        absolute = False
    if not absolute:
        sys.exit(f"ERROR: {path} is not absolute")


def safe_load(nn, weight_path):
    loaded = False
    ntry = 0
    while not loaded:
        try:
            payload = torch.load(weight_path, map_location=nn.device)
            if isinstance(payload, dict) and "model_state_dict" in payload:
                nn.model.load_state_dict(payload["model_state_dict"])
            elif isinstance(payload, dict):
                nn.model.load_state_dict(payload)
            else:
                raise ValueError(f"Unexpected checkpoint payload type: {type(payload)}")
            loaded = True
        except Exception:
            ntry += 1
        if ntry > 1000:
            sys.exit(f"ERROR: could not load {weight_path}")
    return nn


def initialize_params_from_inpt(inpt):
    try:
        seed = int(inpt["seed"])
    except KeyError:
        seed = -1
    try:
        model_id = int(inpt["ID"])
    except KeyError:
        model_id = 0

    epochs = int(inpt["EPOCHS"])
    epochs_lbfgs = int(inpt["EPOCHS_LBFGS"])
    try:
        epochs_start_lbfgs = int(inpt["EPOCHS_START_LBFGS"])
    except KeyError:
        epochs_start_lbfgs = 20

    alpha = [float(entry) for entry in inpt["alpha"].split()]
    learning_rate_weights = float(inpt["LEARNING_RATE_WEIGHTS"])
    learning_rate_weights_final = float(inpt["LEARNING_RATE_WEIGHTS_FINAL"])
    learning_rate_model = float(inpt["LEARNING_RATE_MODEL"])
    learning_rate_model_final = float(inpt["LEARNING_RATE_MODEL_FINAL"])
    learning_rate_lbfgs = float(inpt["LEARNING_RATE_LBFGS"])
    try:
        gradient_threshold = float(inpt["GRADIENT_THRESHOLD"])
    except KeyError:
        gradient_threshold = None

    hard_ic_timescale = float(inpt["HARD_IC_TIMESCALE"])
    ratio_first_time = float(inpt["RATIO_FIRST_TIME"])
    ratio_t_min = float(inpt["RATIO_T_MIN"])
    exp_limiter = float(inpt["EXP_LIMITER"])
    collocation_mode = inpt["COLLOCATION_MODE"]
    gradual_time_sgd = inpt["GRADUAL_TIME_SGD"] == "True"
    gradual_time_lbfgs = inpt["GRADUAL_TIME_LBFGS"] == "True"
    first_time_lbfgs = None
    n_gradual_steps_lbfgs = None
    gradual_time_mode_lbfgs = None
    if gradual_time_lbfgs:
        n_gradual_steps_lbfgs = int(inpt["N_GRADUAL_STEPS_LBFGS"])
        try:
            gradual_time_mode_lbfgs = inpt["GRADUAL_TIME_MODE_LBFGS"]
        except KeyError:
            gradual_time_mode_lbfgs = "linear"

    dynamic_attention_weights = inpt["DYNAMIC_ATTENTION_WEIGHTS"] == "True"
    annealing_weights = inpt["ANNEALING_WEIGHTS"] == "True"
    use_loss_threshold = inpt["USE_LOSS_THRESHOLD"] == "True"
    try:
        loss_threshold = float(inpt["LOSS_THRESHOLD"])
    except KeyError:
        loss_threshold = 1000.0
    try:
        inner_epochs = int(inpt["INNER_EPOCHS"])
    except KeyError:
        inner_epochs = 1
    try:
        start_weight_training_epoch = int(inpt["START_WEIGHT_TRAINING_EPOCH"])
    except KeyError:
        start_weight_training_epoch = 1

    try:
        local_util_folder = _normalize_optional_path(inpt["LOCAL_utilFolder"])
    except KeyError:
        local_util_folder = os.path.join(os.getcwd(), "util")
    try:
        hnn_util_folder = _normalize_optional_path(inpt["HNN_utilFolder"])
    except KeyError:
        hnn_util_folder = None
    try:
        hnn_model_folder = _normalize_optional_path(inpt["HNN_modelFolder"])
    except KeyError:
        hnn_model_folder = None
    try:
        hnn_params = [np.float64(entry) for entry in inpt["HNN_params"].split()]
    except KeyError:
        hnn_params = None
    except ValueError:
        hnn_params = None
    if (
        hnn_util_folder is None
        or hnn_model_folder is None
        or not os.path.isdir(hnn_util_folder)
        or not os.path.isdir(hnn_model_folder)
    ):
        hnn_util_folder = None
        hnn_model_folder = None
        hnn_params = None

    try:
        hnntime_util_folder = _normalize_optional_path(inpt["HNNTIME_utilFolder"])
    except KeyError:
        hnntime_util_folder = None
    try:
        hnntime_model_folder = _normalize_optional_path(inpt["HNNTIME_modelFolder"])
    except KeyError:
        hnntime_model_folder = None
    try:
        hnntime_val = np.float64(inpt["HNNTIME_val"])
    except (KeyError, ValueError):
        hnntime_val = None
    if (
        hnntime_util_folder is None
        or hnntime_model_folder is None
        or hnntime_val is None
        or not os.path.isdir(hnntime_util_folder)
        or not os.path.isdir(hnntime_model_folder)
    ):
        hnntime_util_folder = None
        hnntime_model_folder = None
        hnntime_val = None

    if (hnn_util_folder is not None) or (hnntime_util_folder is not None):
        if not os.path.isdir(local_util_folder):
            print(f"ERROR: {local_util_folder} is not a directory")
            sys.exit()
    absolute_path_check(local_util_folder)
    absolute_path_check(hnn_util_folder)
    absolute_path_check(hnn_model_folder)
    absolute_path_check(hnntime_util_folder)
    absolute_path_check(hnntime_model_folder)

    activation = inpt["ACTIVATION"]
    lbfgs = inpt["LBFGS"] == "True"
    sgd = inpt["SGD"] == "True"
    merged = inpt["MERGED"] == "True"
    linearize_j = inpt["LINEARIZE_J"] == "True"

    try:
        weights = {
            "phie_int": np.float64(inpt["w_phie_int"]),
            "phis_c_int": np.float64(inpt["w_phis_c_int"]),
            "cs_a_int": np.float64(inpt["w_cs_a_int"]),
            "cs_c_int": np.float64(inpt["w_cs_c_int"]),
            "cs_a_rmin_bound": np.float64(inpt["w_cs_a_rmin_bound"]),
            "cs_a_rmax_bound": np.float64(inpt["w_cs_a_rmax_bound"]),
            "cs_c_rmin_bound": np.float64(inpt["w_cs_c_rmin_bound"]),
            "cs_c_rmax_bound": np.float64(inpt["w_cs_c_rmax_bound"]),
            "phie_dat": np.float64(inpt["w_phie_dat"]),
            "phis_c_dat": np.float64(inpt["w_phis_c_dat"]),
            "cs_a_dat": np.float64(inpt["w_cs_a_dat"]),
            "cs_c_dat": np.float64(inpt["w_cs_c_dat"]),
        }
    except KeyError:
        weights = None

    batch_size_int = int(inpt["BATCH_SIZE_INT"])
    batch_size_bound = int(inpt["BATCH_SIZE_BOUND"])
    max_batch_size_data = int(inpt["MAX_BATCH_SIZE_DATA"])
    batch_size_reg = int(inpt["BATCH_SIZE_REG"])
    n_batch = int(inpt["N_BATCH"])
    n_batch_lbfgs = int(inpt["N_BATCH_LBFGS"])
    neurons_num = int(inpt["NEURONS_NUM"])
    layers_t_num = int(inpt["LAYERS_T_NUM"])
    layers_tr_num = int(inpt["LAYERS_TR_NUM"])
    layers_t_var_num = int(inpt["LAYERS_T_VAR_NUM"])
    layers_tr_var_num = int(inpt["LAYERS_TR_VAR_NUM"])
    layers_split_num = int(inpt["LAYERS_SPLIT_NUM"])
    try:
        num_res_blocks = int(inpt["NUM_RES_BLOCKS"])
    except Exception:
        num_res_blocks = 0
    if num_res_blocks > 0:
        num_res_block_layers = int(inpt["NUM_RES_BLOCK_LAYERS"])
        num_res_block_units = int(inpt["NUM_RES_BLOCK_UNITS"])
    else:
        num_res_block_layers = 1
        num_res_block_units = 1
    try:
        num_grad_path_layers = int(inpt["NUM_GRAD_PATH_LAYERS"])
    except Exception:
        num_grad_path_layers = None
    if num_grad_path_layers is not None:
        num_grad_path_units = int(inpt["NUM_GRAD_PATH_UNITS"])
    else:
        num_grad_path_units = None

    try:
        load_model = _normalize_optional_path(inpt["LOAD_MODEL"])
        if load_model is not None and not os.path.isfile(load_model):
            load_model = None
    except KeyError:
        load_model = None

    return {
        "MERGED": merged,
        "NEURONS_NUM": neurons_num,
        "LAYERS_T_NUM": layers_t_num,
        "LAYERS_TR_NUM": layers_tr_num,
        "LAYERS_T_VAR_NUM": layers_t_var_num,
        "LAYERS_TR_VAR_NUM": layers_tr_var_num,
        "LAYERS_SPLIT_NUM": layers_split_num,
        "seed": seed,
        "ID": model_id,
        "EPOCHS": epochs,
        "EPOCHS_LBFGS": epochs_lbfgs,
        "EPOCHS_START_LBFGS": epochs_start_lbfgs,
        "alpha": alpha,
        "LEARNING_RATE_WEIGHTS": learning_rate_weights,
        "LEARNING_RATE_WEIGHTS_FINAL": learning_rate_weights_final,
        "LEARNING_RATE_MODEL": learning_rate_model,
        "LEARNING_RATE_MODEL_FINAL": learning_rate_model_final,
        "LEARNING_RATE_LBFGS": learning_rate_lbfgs,
        "GRADIENT_THRESHOLD": gradient_threshold,
        "HARD_IC_TIMESCALE": hard_ic_timescale,
        "RATIO_FIRST_TIME": ratio_first_time,
        "RATIO_T_MIN": ratio_t_min,
        "EXP_LIMITER": exp_limiter,
        "COLLOCATION_MODE": collocation_mode,
        "GRADUAL_TIME_SGD": gradual_time_sgd,
        "GRADUAL_TIME_LBFGS": gradual_time_lbfgs,
        "FIRST_TIME_LBFGS": first_time_lbfgs,
        "N_GRADUAL_STEPS_LBFGS": n_gradual_steps_lbfgs,
        "GRADUAL_TIME_MODE_LBFGS": gradual_time_mode_lbfgs,
        "DYNAMIC_ATTENTION_WEIGHTS": dynamic_attention_weights,
        "ANNEALING_WEIGHTS": annealing_weights,
        "USE_LOSS_THRESHOLD": use_loss_threshold,
        "LOSS_THRESHOLD": loss_threshold,
        "INNER_EPOCHS": inner_epochs,
        "START_WEIGHT_TRAINING_EPOCH": start_weight_training_epoch,
        "HNN_utilFolder": hnn_util_folder,
        "HNN_modelFolder": hnn_model_folder,
        "HNN_params": hnn_params,
        "HNNTIME_utilFolder": hnntime_util_folder,
        "HNNTIME_modelFolder": hnntime_model_folder,
        "HNNTIME_val": hnntime_val,
        "ACTIVATION": activation,
        "LBFGS": lbfgs,
        "SGD": sgd,
        "LINEARIZE_J": linearize_j,
        "weights": weights,
        "BATCH_SIZE_INT": batch_size_int,
        "BATCH_SIZE_BOUND": batch_size_bound,
        "MAX_BATCH_SIZE_DATA": max_batch_size_data,
        "BATCH_SIZE_REG": batch_size_reg,
        "N_BATCH": n_batch,
        "N_BATCH_LBFGS": n_batch_lbfgs,
        "NUM_RES_BLOCKS": num_res_blocks,
        "NUM_RES_BLOCK_LAYERS": num_res_block_layers,
        "NUM_RES_BLOCK_UNITS": num_res_block_units,
        "NUM_GRAD_PATH_LAYERS": num_grad_path_layers,
        "NUM_GRAD_PATH_UNITS": num_grad_path_units,
        "LOAD_MODEL": load_model,
        "LOCAL_utilFolder": local_util_folder,
    }


def initialize_params(args):
    inpt = parse_input_file(args.input_file)
    return initialize_params_from_inpt(inpt)


def initialize_nn(args, input_params):
    seed = input_params["seed"]
    neurons_num = input_params["NEURONS_NUM"]
    layers_t_num = input_params["LAYERS_T_NUM"]
    layers_tr_num = input_params["LAYERS_TR_NUM"]
    layers_t_var_num = input_params["LAYERS_T_VAR_NUM"]
    layers_tr_var_num = input_params["LAYERS_TR_VAR_NUM"]
    layers_split_num = input_params["LAYERS_SPLIT_NUM"]
    alpha = input_params["alpha"]
    n_batch = input_params["N_BATCH"]
    epochs = input_params["EPOCHS"]
    num_res_blocks = input_params["NUM_RES_BLOCKS"]
    num_res_block_layers = input_params["NUM_RES_BLOCK_LAYERS"]
    num_res_block_units = input_params["NUM_RES_BLOCK_UNITS"]
    num_grad_path_layers = input_params["NUM_GRAD_PATH_LAYERS"]
    num_grad_path_units = input_params["NUM_GRAD_PATH_UNITS"]
    batch_size_int = input_params["BATCH_SIZE_INT"]
    batch_size_bound = input_params["BATCH_SIZE_BOUND"]
    batch_size_reg = input_params["BATCH_SIZE_REG"]
    max_batch_size_data = input_params["MAX_BATCH_SIZE_DATA"]
    n_batch_lbfgs = input_params["N_BATCH_LBFGS"]
    hard_ic_timescale = input_params["HARD_IC_TIMESCALE"]
    exp_limiter = input_params["EXP_LIMITER"]
    collocation_mode = input_params["COLLOCATION_MODE"]
    gradual_time_sgd = input_params["GRADUAL_TIME_SGD"]
    gradual_time_lbfgs = input_params["GRADUAL_TIME_LBFGS"]
    gradual_time_mode_lbfgs = input_params["GRADUAL_TIME_MODE_LBFGS"]
    ratio_first_time = input_params["RATIO_FIRST_TIME"]
    n_gradual_steps_lbfgs = input_params["N_GRADUAL_STEPS_LBFGS"]
    ratio_t_min = input_params["RATIO_T_MIN"]
    epochs_lbfgs = input_params["EPOCHS_LBFGS"]
    epochs_start_lbfgs = input_params["EPOCHS_START_LBFGS"]
    loss_threshold = input_params["LOSS_THRESHOLD"]
    dynamic_attention_weights = input_params["DYNAMIC_ATTENTION_WEIGHTS"]
    annealing_weights = input_params["ANNEALING_WEIGHTS"]
    use_loss_threshold = input_params["USE_LOSS_THRESHOLD"]
    activation = input_params["ACTIVATION"]
    lbfgs = input_params["LBFGS"]
    sgd = input_params["SGD"]
    linearize_j = input_params["LINEARIZE_J"]
    load_model = input_params["LOAD_MODEL"]
    merged = input_params["MERGED"]
    model_id = input_params["ID"]
    local_util_folder = input_params["LOCAL_utilFolder"]
    hnn_model_folder = input_params["HNN_modelFolder"]
    hnn_util_folder = input_params["HNN_utilFolder"]
    hnn_params = input_params["HNN_params"]
    hnntime_model_folder = input_params["HNNTIME_modelFolder"]
    hnntime_util_folder = input_params["HNNTIME_utilFolder"]
    hnntime_val = input_params["HNNTIME_val"]
    weights = input_params["weights"]

    if args.simpleModel:
        from spm_simpler import makeParams
    else:
        from spm import makeParams

    params = makeParams()
    dataFolder = args.dataFolder

    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if merged:
        hidden_units_t = [neurons_num] * layers_t_num
        hidden_units_t_r = [neurons_num] * layers_tr_num
        hidden_units_cs_a = [neurons_num] * layers_tr_var_num
        hidden_units_cs_c = [neurons_num] * layers_tr_var_num
        hidden_units_phie = [neurons_num] * layers_t_var_num
        hidden_units_phis_c = [neurons_num] * layers_t_var_num
    else:
        hidden_units_t = None
        hidden_units_t_r = None
        hidden_units_cs_a = [neurons_num] * layers_split_num
        hidden_units_cs_c = [neurons_num] * layers_split_num
        hidden_units_phie = [neurons_num] * layers_split_num
        hidden_units_phis_c = [neurons_num] * layers_split_num

    if dataFolder is not None and os.path.isdir(dataFolder) and alpha[2] > 0:
        try:
            data_phie = np.load(os.path.join(dataFolder, "data_phie_multi.npz"))
            use_multi = True
            print("INFO: LOADING MULTI DATASETS")
        except Exception:
            data_phie = np.load(os.path.join(dataFolder, "data_phie.npz"))
            use_multi = False
            print("INFO: LOADING SINGLE DATASETS")

        xTrain_phie = data_phie["x_train"].astype("float64")
        yTrain_phie = data_phie["y_train"].astype("float64")
        x_params_train_phie = data_phie["x_params_train"].astype("float64")

        if use_multi:
            data_phis_c = np.load(os.path.join(dataFolder, "data_phis_c_multi.npz"))
        else:
            data_phis_c = np.load(os.path.join(dataFolder, "data_phis_c.npz"))
        xTrain_phis_c = data_phis_c["x_train"].astype("float64")
        yTrain_phis_c = data_phis_c["y_train"].astype("float64")
        x_params_train_phis_c = data_phis_c["x_params_train"].astype("float64")

        if use_multi:
            data_cs_a = np.load(os.path.join(dataFolder, "data_cs_a_multi.npz"))
        else:
            data_cs_a = np.load(os.path.join(dataFolder, "data_cs_a.npz"))
        xTrain_cs_a = data_cs_a["x_train"].astype("float64")
        yTrain_cs_a = data_cs_a["y_train"].astype("float64")
        x_params_train_cs_a = data_cs_a["x_params_train"].astype("float64")

        if use_multi:
            data_cs_c = np.load(os.path.join(dataFolder, "data_cs_c_multi.npz"))
        else:
            data_cs_c = np.load(os.path.join(dataFolder, "data_cs_c.npz"))
        xTrain_cs_c = data_cs_c["x_train"].astype("float64")
        yTrain_cs_c = data_cs_c["y_train"].astype("float64")
        x_params_train_cs_c = data_cs_c["x_params_train"].astype("float64")
    else:
        nParams = 2
        print("INFO: LOADING DUMMY DATA")
        xTrain_phie = np.zeros((n_batch, 1)).astype("float64")
        yTrain_phie = np.zeros((n_batch, 1)).astype("float64")
        x_params_train_phie = np.zeros((n_batch, nParams)).astype("float64")
        xTrain_phis_c = np.zeros((n_batch, 1)).astype("float64")
        yTrain_phis_c = np.zeros((n_batch, 1)).astype("float64")
        x_params_train_phis_c = np.zeros((n_batch, nParams)).astype("float64")
        xTrain_cs_a = np.zeros((n_batch, 2)).astype("float64")
        yTrain_cs_a = np.zeros((n_batch, 1)).astype("float64")
        x_params_train_cs_a = np.zeros((n_batch, nParams)).astype("float64")
        xTrain_cs_c = np.zeros((n_batch, 2)).astype("float64")
        yTrain_cs_c = np.zeros((n_batch, 1)).astype("float64")
        x_params_train_cs_c = np.zeros((n_batch, nParams)).astype("float64")

    nn = myNN(
        params=params,
        hidden_units_t=hidden_units_t,
        hidden_units_t_r=hidden_units_t_r,
        hidden_units_phie=hidden_units_phie,
        hidden_units_phis_c=hidden_units_phis_c,
        hidden_units_cs_a=hidden_units_cs_a,
        hidden_units_cs_c=hidden_units_cs_c,
        n_hidden_res_blocks=num_res_blocks,
        n_res_block_layers=num_res_block_layers,
        n_res_block_units=num_res_block_units,
        n_grad_path_layers=num_grad_path_layers,
        n_grad_path_units=num_grad_path_units,
        alpha=alpha,
        batch_size_int=batch_size_int,
        batch_size_bound=batch_size_bound,
        batch_size_reg=batch_size_reg,
        max_batch_size_data=max_batch_size_data,
        n_batch=n_batch,
        n_batch_lbfgs=n_batch_lbfgs,
        hard_IC_timescale=np.float64(hard_ic_timescale),
        exponentialLimiter=exp_limiter,
        collocationMode=collocation_mode,
        gradualTime_sgd=gradual_time_sgd,
        gradualTime_lbfgs=gradual_time_lbfgs,
        gradualTimeMode_lbfgs=gradual_time_mode_lbfgs,
        firstTime=np.float64(hard_ic_timescale * ratio_first_time),
        n_gradual_steps_lbfgs=n_gradual_steps_lbfgs,
        tmin_int_bound=np.float64(hard_ic_timescale * ratio_t_min),
        nEpochs=epochs,
        nEpochs_lbfgs=epochs_lbfgs,
        nEpochs_start_lbfgs=epochs_start_lbfgs,
        initialLossThreshold=np.float64(loss_threshold),
        dynamicAttentionWeights=dynamic_attention_weights,
        annealingWeights=annealing_weights,
        useLossThreshold=use_loss_threshold,
        activation=activation,
        lbfgs=lbfgs,
        sgd=sgd,
        linearizeJ=linearize_j,
        params_min=[params["deg_i0_a_min"], params["deg_ds_c_min"]],
        params_max=[params["deg_i0_a_max"], params["deg_ds_c_max"]],
        xDataList=[xTrain_phie, xTrain_phis_c, xTrain_cs_a, xTrain_cs_c],
        x_params_dataList=[
            x_params_train_phie,
            x_params_train_phis_c,
            x_params_train_cs_a,
            x_params_train_cs_c,
        ],
        yDataList=[yTrain_phie, yTrain_phis_c, yTrain_cs_a, yTrain_cs_c],
        logLossFolder="Log_" + str(model_id),
        modelFolder="Model_" + str(model_id),
        local_utilFolder=local_util_folder,
        hnn_utilFolder=hnn_util_folder,
        hnn_modelFolder=hnn_model_folder,
        hnn_params=hnn_params,
        hnntime_utilFolder=hnntime_util_folder,
        hnntime_modelFolder=hnntime_model_folder,
        hnntime_val=hnntime_val,
        weights=weights,
        verbose=True,
    )

    if not load_model is None:
        print(f"INFO: Loading model {load_model}")
        nn = safe_load(nn, load_model)

    if not args.optimized:
        print("INFO: PyTorch patch loaded. Keras model plotting is skipped in this version.")

    return nn


def initialize_nn_from_params_config(params, configDict):
    hidden_units_t = configDict["hidden_units_t"]
    hidden_units_t_r = configDict["hidden_units_t_r"]
    hidden_units_phie = configDict["hidden_units_phie"]
    hidden_units_phis_c = configDict["hidden_units_phis_c"]
    hidden_units_cs_a = configDict["hidden_units_cs_a"]
    hidden_units_cs_c = configDict["hidden_units_cs_c"]
    try:
        n_hidden_res_blocks = configDict["n_hidden_res_blocks"]
    except Exception:
        n_hidden_res_blocks = 0
    if n_hidden_res_blocks > 0:
        n_res_block_layers = configDict["n_res_block_layers"]
        n_res_block_units = configDict["n_res_block_units"]
    else:
        n_res_block_layers = 1
        n_res_block_units = 1
    try:
        n_grad_path_layers = configDict["n_grad_path_layers"]
    except Exception:
        n_grad_path_layers = None
    if n_grad_path_layers is not None and n_grad_path_layers > 0:
        n_grad_path_units = configDict["n_grad_path_units"]
    else:
        n_grad_path_units = None
    hard_ic_timescale = configDict["hard_IC_timescale"]
    exp_limiter = configDict["exponentialLimiter"]
    activation = configDict["activation"]
    try:
        linearize_j = configDict["linearizeJ"]
    except Exception:
        linearize_j = True
    try:
        dynamic_attention = configDict["dynamicAttentionWeights"]
    except Exception:
        dynamic_attention = False
    try:
        annealing_weights = configDict["annealingWeights"]
    except Exception:
        annealing_weights = False
    activeInt = configDict.get("activeInt", True)
    activeBound = configDict.get("activeBound", True)
    activeData = configDict.get("activeData", False)
    activeReg = configDict.get("activeReg", False)
    try:
        params_min = configDict["params_min"]
    except Exception:
        params_min = [params["deg_i0_a_min"], params["deg_ds_c_min"]]
    try:
        params_max = configDict["params_max"]
    except Exception:
        params_max = [params["deg_i0_a_max"], params["deg_ds_c_max"]]
    local_utilFolder = configDict.get("local_utilFolder", None)
    hnn_utilFolder = configDict.get("hnn_utilFolder", None)
    hnn_modelFolder = configDict.get("hnn_modelFolder", None)
    hnn_params = configDict.get("hnn_params", None)
    hnntime_utilFolder = configDict.get("hnntime_utilFolder", None)
    hnntime_modelFolder = configDict.get("hnntime_modelFolder", None)
    hnntime_val = configDict.get("hnntime_val", None)

    nn = myNN(
        params=params,
        hidden_units_t=hidden_units_t,
        hidden_units_t_r=hidden_units_t_r,
        hidden_units_phie=hidden_units_phie,
        hidden_units_phis_c=hidden_units_phis_c,
        hidden_units_cs_a=hidden_units_cs_a,
        hidden_units_cs_c=hidden_units_cs_c,
        n_hidden_res_blocks=n_hidden_res_blocks,
        n_res_block_layers=n_res_block_layers,
        n_res_block_units=n_res_block_units,
        n_grad_path_layers=n_grad_path_layers,
        n_grad_path_units=n_grad_path_units,
        hard_IC_timescale=np.float64(hard_ic_timescale),
        exponentialLimiter=exp_limiter,
        dynamicAttentionWeights=dynamic_attention,
        annealingWeights=annealing_weights,
        activation=activation,
        linearizeJ=linearize_j,
        params_min=params_min,
        params_max=params_max,
        local_utilFolder=local_utilFolder,
        hnn_utilFolder=hnn_utilFolder,
        hnn_modelFolder=hnn_modelFolder,
        hnn_params=hnn_params,
        hnntime_utilFolder=hnntime_utilFolder,
        hnntime_modelFolder=hnntime_modelFolder,
        hnntime_val=hnntime_val,
        verbose=True,
    )

    return nn
