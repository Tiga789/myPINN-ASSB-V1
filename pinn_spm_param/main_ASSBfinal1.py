import math
import os
import shutil
import sys
import time

import numpy as np

sys.path.append("util")

import argument
from init_pinn_ASSBfinal1 import initialize_nn, initialize_params, initialize_params_from_inpt



def _copytree_overwrite(src: str, dst: str) -> None:
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)



def do_training_only(input_params, nn):
    learning_rate_lbfgs = input_params["LEARNING_RATE_LBFGS"]
    learning_rate_model = input_params["LEARNING_RATE_MODEL"]
    learning_rate_model_final = input_params["LEARNING_RATE_MODEL_FINAL"]
    learning_rate_weights = input_params["LEARNING_RATE_WEIGHTS"]
    learning_rate_weights_final = input_params["LEARNING_RATE_WEIGHTS_FINAL"]
    gradient_threshold = input_params["GRADIENT_THRESHOLD"]
    inner_epochs = input_params["INNER_EPOCHS"]
    epochs = input_params["EPOCHS"]
    start_weight_training_epoch = input_params["START_WEIGHT_TRAINING_EPOCH"]

    factor_scheduler_model = np.log(
        learning_rate_model_final / (learning_rate_model + 1e-16)
    ) / ((epochs + 1e-16) / 2.0)

    def scheduler_model(epoch, lr):
        if epoch < epochs // 2:
            return lr
        return max(lr * math.exp(float(factor_scheduler_model)), learning_rate_model_final)

    factor_scheduler_weights = np.log(
        learning_rate_weights_final / (learning_rate_weights + 1e-16)
    ) / ((epochs + 1e-16) / 2.0)

    def scheduler_weights(epoch, lr):
        if epoch < epochs // 2:
            return lr
        return max(lr * math.exp(float(factor_scheduler_weights)), learning_rate_weights_final)

    time_start = time.time()
    unweighted_loss = nn.train(
        learningRateModel=learning_rate_model,
        learningRateModelFinal=learning_rate_model_final,
        lrSchedulerModel=scheduler_model,
        learningRateWeights=learning_rate_weights,
        learningRateWeightsFinal=learning_rate_weights_final,
        lrSchedulerWeights=scheduler_weights,
        learningRateLBFGS=learning_rate_lbfgs,
        inner_epochs=inner_epochs,
        start_weight_training_epoch=start_weight_training_epoch,
        gradient_threshold=gradient_threshold,
    )
    time_end = time.time()
    return time_end - time_start, unweighted_loss



def do_training(input_params, nn):
    model_id = input_params["ID"]
    elapsed_time, unweighted_loss = do_training_only(input_params, nn)
    _copytree_overwrite(nn.modelFolder, "ModelFin_" + str(model_id))
    _copytree_overwrite(nn.logLossFolder, "LogFin_" + str(model_id))
    return elapsed_time, unweighted_loss



def main():
    args = argument.initArg()
    input_params = initialize_params(args)
    nn = initialize_nn(args=args, input_params=input_params)
    elapsed, unweighted_loss = do_training(input_params=input_params, nn=nn)
    print(f"Total time {elapsed:.2f}s")
    print(f"Unweighted loss {unweighted_loss}")


if __name__ == "__main__":
    main()
