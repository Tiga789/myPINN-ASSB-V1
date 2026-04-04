import json
import os
from importlib.machinery import SourceFileLoader

import argument

args = argument.initArg()


def _load_module(module_name: str, module_path: str):
    return SourceFileLoader(module_name, module_path).load_module()


def reload(utilFolder, localUtilFolder, params_loaded, checkRescale=False):
    _load_module("_losses", os.path.join(localUtilFolder, "_losses.py"))
    _load_module("_rescale", os.path.join(localUtilFolder, "_rescale.py"))
    _load_module("uocp_cs", os.path.join(localUtilFolder, "uocp_cs.py"))
    _load_module("thermo", os.path.join(localUtilFolder, "thermo.py"))
    if args.simpleModel:
        _load_module("spm_simpler", os.path.join(localUtilFolder, "spm_simpler.py"))
    else:
        _load_module("spm", os.path.join(localUtilFolder, "spm.py"))
    _load_module("myNN", os.path.join(localUtilFolder, "myNN.py"))
    _load_module("init_pinn", os.path.join(localUtilFolder, "init_pinn.py"))


def load_model(utilFolder, modelFolder, localUtilFolder, loadDep=False, checkRescale=False):
    _load_module("_losses", os.path.join(utilFolder, "_losses.py"))
    _load_module("_rescale", os.path.join(utilFolder, "_rescale.py"))
    _load_module("uocp_cs", os.path.join(utilFolder, "uocp_cs.py"))
    _load_module("thermo", os.path.join(utilFolder, "thermo.py"))
    if args.simpleModel:
        spm_simpler = _load_module("spm_simpler", os.path.join(utilFolder, "spm_simpler.py"))
        makeParams = spm_simpler.makeParams
    else:
        spm = _load_module("spm", os.path.join(utilFolder, "spm.py"))
        makeParams = spm.makeParams
    _load_module("myNN", os.path.join(utilFolder, "myNN.py"))
    init_pinn = _load_module("init_pinn", os.path.join(localUtilFolder, "init_pinn.py"))

    params_loaded = makeParams()

    with open(os.path.join(modelFolder, "config.json"), "r", encoding="utf-8") as json_file:
        configDict = json.load(json_file)
    nn = init_pinn.initialize_nn_from_params_config(params_loaded, configDict)

    weight_path = os.path.join(modelFolder, "best.weights.h5")
    if not os.path.isfile(weight_path):
        alt = os.path.join(modelFolder, "best.pt")
        if os.path.isfile(alt):
            weight_path = alt
        else:
            alt = os.path.join(modelFolder, "last.weights.h5")
            if os.path.isfile(alt):
                weight_path = alt
    nn = init_pinn.safe_load(nn, weight_path)

    reload(utilFolder, localUtilFolder, params_loaded, checkRescale=checkRescale)
    return nn
