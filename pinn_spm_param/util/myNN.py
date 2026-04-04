import json
import math
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from custom_activations import swish_activation
from dataTools import checkDataShape, completeDataset
from torch_utils import (
    ActivationModule,
    GradientPathBlock,
    PreReshape,
    ResidualBlock,
    default_device,
    ensure_2d,
    kaiming_init,
    to_tensor,
)


torch.set_default_dtype(torch.float64)


def safe_save(model: nn.Module, weight_path: str, overwrite: bool = False) -> None:
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    torch.save(payload, weight_path)


class DenseStack(nn.Module):
    def __init__(self, in_features: int, hidden_units, activation: str):
        super().__init__()
        hidden_units = list(hidden_units or [])
        layers: list[nn.Module] = []
        current = int(in_features)
        for width in hidden_units:
            layers.append(nn.Linear(current, int(width)))
            layers.append(ActivationModule(activation))
            current = int(width)
        self.net = nn.Sequential(*layers)
        self.out_features = current
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBranchHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_units,
        activation: str,
        n_hidden_res_blocks: int,
        n_res_block_layers: int,
        n_res_block_units: int,
        use_bias: bool = True,
    ):
        super().__init__()
        hidden_units = list(hidden_units or [])
        self.pre_dense = DenseStack(in_features, hidden_units, activation)
        last_unit = self.pre_dense.out_features
        self.use_res = int(n_hidden_res_blocks) > 0
        if self.use_res:
            self.pre_res = PreReshape(last_unit, int(n_res_block_units), activation)
            self.res_blocks = nn.ModuleList(
                [
                    ResidualBlock(int(n_res_block_units), int(n_res_block_layers), activation)
                    for _ in range(int(n_hidden_res_blocks))
                ]
            )
            final_in = int(n_res_block_units)
        else:
            self.pre_res = nn.Identity()
            self.res_blocks = nn.ModuleList([])
            final_in = last_unit
        self.output = nn.Linear(final_in, 1, bias=use_bias)
        self.output.apply(kaiming_init)
        if not use_bias and self.output.bias is not None:
            self.output.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_dense(x)
        x = self.pre_res(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output(x)


class GradientPathHead(nn.Module):
    def __init__(self, in_features: int, n_layers: int, n_units: int, activation: str, use_bias: bool = True):
        super().__init__()
        self.grad_path = GradientPathBlock(int(in_features), int(n_layers), int(n_units), activation)
        self.output = nn.Linear(int(n_units), 1, bias=use_bias)
        self.output.apply(kaiming_init)
        if not use_bias and self.output.bias is not None:
            self.output.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grad_path(x)
        return self.output(x)


class TorchPINNModel(nn.Module):
    def __init__(self, owner: "myNN"):
        super().__init__()
        self.owner = owner
        act = owner.activation

        self.use_grad_path = owner.n_grad_path_layers is not None and int(owner.n_grad_path_layers) > 0
        self.is_merged = owner.hidden_units_t is not None

        if self.use_grad_path:
            self.t_stack = DenseStack(3, owner.hidden_units_t, act)
            self.tr_stack = DenseStack(self.t_stack.out_features + 1, owner.hidden_units_t_r, act)
            self.phie_head = GradientPathHead(self.t_stack.out_features, owner.n_grad_path_layers, owner.n_grad_path_units, act, use_bias=True)
            self.phis_c_head = GradientPathHead(self.t_stack.out_features, owner.n_grad_path_layers, owner.n_grad_path_units, act, use_bias=True)
            self.cs_a_head = GradientPathHead(self.tr_stack.out_features, owner.n_grad_path_layers, owner.n_grad_path_units, act, use_bias=False)
            self.cs_c_head = GradientPathHead(self.tr_stack.out_features, owner.n_grad_path_layers, owner.n_grad_path_units, act, use_bias=False)
        elif self.is_merged:
            self.t_stack = DenseStack(3, owner.hidden_units_t, act)
            self.tr_stack = DenseStack(self.t_stack.out_features + 1, owner.hidden_units_t_r, act)
            self.phie_head = ResBranchHead(
                self.t_stack.out_features,
                owner.hidden_units_phie,
                act,
                owner.n_hidden_res_blocks,
                owner.n_res_block_layers,
                owner.n_res_block_units,
                use_bias=True,
            )
            self.phis_c_head = ResBranchHead(
                self.t_stack.out_features,
                owner.hidden_units_phis_c,
                act,
                owner.n_hidden_res_blocks,
                owner.n_res_block_layers,
                owner.n_res_block_units,
                use_bias=True,
            )
            self.cs_a_head = ResBranchHead(
                self.tr_stack.out_features,
                owner.hidden_units_cs_a,
                act,
                owner.n_hidden_res_blocks,
                owner.n_res_block_layers,
                owner.n_res_block_units,
                use_bias=False,
            )
            self.cs_c_head = ResBranchHead(
                self.tr_stack.out_features,
                owner.hidden_units_cs_c,
                act,
                owner.n_hidden_res_blocks,
                owner.n_res_block_layers,
                owner.n_res_block_units,
                use_bias=False,
            )
        else:
            self.phie_head = ResBranchHead(
                3,
                owner.hidden_units_phie,
                act,
                owner.n_hidden_res_blocks,
                owner.n_res_block_layers,
                owner.n_res_block_units,
                use_bias=True,
            )
            self.phis_c_head = ResBranchHead(
                3,
                owner.hidden_units_phis_c,
                act,
                owner.n_hidden_res_blocks,
                owner.n_res_block_layers,
                owner.n_res_block_units,
                use_bias=True,
            )
            self.cs_a_head = ResBranchHead(
                4,
                owner.hidden_units_cs_a,
                act,
                owner.n_hidden_res_blocks,
                owner.n_res_block_layers,
                owner.n_res_block_units,
                use_bias=False,
            )
            self.cs_c_head = ResBranchHead(
                4,
                owner.hidden_units_cs_c,
                act,
                owner.n_hidden_res_blocks,
                owner.n_res_block_layers,
                owner.n_res_block_units,
                use_bias=False,
            )

    def _prepare_inputs(self, inputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 4:
            raise ValueError("Model expects a list/tuple [t, r, deg_i0_a, deg_ds_c].")
        t = ensure_2d(inputs[0], device=self.owner.device)
        r = ensure_2d(inputs[1], device=self.owner.device)
        deg_i0_a = ensure_2d(inputs[2], device=self.owner.device)
        deg_ds_c = ensure_2d(inputs[3], device=self.owner.device)
        return t, r, deg_i0_a, deg_ds_c

    def forward(self, inputs, training: bool = False):
        if training:
            super().train(True)
        t, r, deg_i0_a, deg_ds_c = self._prepare_inputs(inputs)

        t_par = torch.cat([t, deg_i0_a, deg_ds_c], dim=1)

        if self.use_grad_path:
            tmp_t = self.t_stack(t_par)
            tmp_t_r = self.tr_stack(torch.cat([tmp_t, r], dim=1))
            output_phie = self.phie_head(tmp_t)
            output_phis_c = self.phis_c_head(tmp_t)
            output_cs_a = self.cs_a_head(tmp_t_r)
            output_cs_c = self.cs_c_head(tmp_t_r)
        elif self.is_merged:
            tmp_t = self.t_stack(t_par)
            tmp_t_r = self.tr_stack(torch.cat([tmp_t, r], dim=1))
            output_phie = self.phie_head(tmp_t)
            output_phis_c = self.phis_c_head(tmp_t)
            output_cs_a = self.cs_a_head(tmp_t_r)
            output_cs_c = self.cs_c_head(tmp_t_r)
        else:
            output_phie = self.phie_head(t_par)
            output_phis_c = self.phis_c_head(t_par)
            tmp_cs = torch.cat([t_par, r], dim=1)
            output_cs_a = self.cs_a_head(tmp_cs)
            output_cs_c = self.cs_c_head(tmp_cs)

        return [output_phie, output_phis_c, output_cs_a, output_cs_c]


class myNN:
    def __init__(
        self,
        params,
        hidden_units_t=None,
        hidden_units_t_r=None,
        hidden_units_phie=None,
        hidden_units_phis_c=None,
        hidden_units_cs_a=None,
        hidden_units_cs_c=None,
        n_hidden_res_blocks=0,
        n_res_block_layers=1,
        n_res_block_units=1,
        n_grad_path_layers=None,
        n_grad_path_units=None,
        alpha=[0, 0, 0, 0],
        batch_size_int=0,
        batch_size_bound=0,
        max_batch_size_data=0,
        batch_size_reg=0,
        batch_size_struct=64,
        n_batch=0,
        n_batch_lbfgs=0,
        nEpochs_start_lbfgs=10,
        hard_IC_timescale=np.float64(0.81),
        exponentialLimiter=np.float64(10.0),
        collocationMode="fixed",
        gradualTime_sgd=False,
        gradualTime_lbfgs=False,
        firstTime=np.float64(0.1),
        n_gradual_steps_lbfgs=None,
        gradualTimeMode_lbfgs=None,
        tmin_int_bound=np.float64(0.1),
        nEpochs=60,
        nEpochs_lbfgs=60,
        initialLossThreshold=np.float64(100),
        dynamicAttentionWeights=False,
        annealingWeights=False,
        useLossThreshold=True,
        activation="tanh",
        linearizeJ=False,
        lbfgs=False,
        sgd=True,
        params_max=[],
        params_min=[],
        xDataList=[],
        x_params_dataList=[],
        yDataList=[],
        logLossFolder=None,
        modelFolder=None,
        local_utilFolder=None,
        hnn_utilFolder=None,
        hnn_modelFolder=None,
        hnn_params=None,
        hnntime_utilFolder=None,
        hnntime_modelFolder=None,
        hnntime_val=None,
        verbose=False,
        weights=None,
    ):
        self.verbose = verbose
        self.device = default_device()

        self.logLossFolder = "Log" if logLossFolder is None else logLossFolder
        self.modelFolder = "Model" if modelFolder is None else modelFolder
        self.params = params
        self.local_utilFolder = local_utilFolder
        self.hnn_utilFolder = hnn_utilFolder
        self.hnn_modelFolder = hnn_modelFolder
        self.hnn_params = hnn_params
        self.hnntime_utilFolder = hnntime_utilFolder
        self.hnntime_modelFolder = hnntime_modelFolder
        self.hnntime_val = hnntime_val

        self.use_hnn = False
        self.use_hnntime = False
        if hnn_utilFolder is not None or hnn_modelFolder is not None:
            raise NotImplementedError("HNN loading is not yet ported in this first PyTorch patch.")
        if hnntime_utilFolder is not None or hnntime_modelFolder is not None:
            raise NotImplementedError("HNNTIME loading is not yet ported in this first PyTorch patch.")
        if dynamicAttentionWeights:
            raise NotImplementedError("Dynamic attention weights are not yet ported in this first PyTorch patch.")
        if annealingWeights:
            raise NotImplementedError("Annealing weights are not yet ported in this first PyTorch patch.")

        self.hidden_units_t = hidden_units_t
        self.hidden_units_t_r = hidden_units_t_r
        self.hidden_units_phie = hidden_units_phie
        self.hidden_units_phis_c = hidden_units_phis_c
        self.hidden_units_cs_a = hidden_units_cs_a
        self.hidden_units_cs_c = hidden_units_cs_c
        self.n_hidden_res_blocks = int(n_hidden_res_blocks)
        self.n_res_block_layers = int(n_res_block_layers)
        self.n_res_block_units = int(n_res_block_units)
        self.n_grad_path_layers = None if n_grad_path_layers is None else int(n_grad_path_layers)
        self.n_grad_path_units = None if n_grad_path_units is None else int(n_grad_path_units)
        self.dynamicAttentionWeights = bool(dynamicAttentionWeights)
        self.annealingWeights = bool(annealingWeights)
        self.useLossThreshold = bool(useLossThreshold)
        self.activation = activation.lower()
        self.tmin = np.float64(self.params["tmin"])
        self.tmax = np.float64(self.params["tmax"])
        self.rmin = np.float64(self.params["rmin"])
        self.rmax_a = np.float64(self.params["Rs_a"])
        self.rmax_c = np.float64(self.params["Rs_c"])

        self.ind_t = np.int32(0)
        self.ind_r = np.int32(1)
        self.ind_phie = np.int32(0)
        self.ind_phis_c = np.int32(1)
        self.ind_cs_offset = np.int32(2)
        self.ind_cs_a = np.int32(2)
        self.ind_cs_c = np.int32(3)

        self.ind_phie_data = np.int32(0)
        self.ind_phis_c_data = np.int32(1)
        self.ind_cs_offset_data = np.int32(2)
        self.ind_cs_a_data = np.int32(2)
        self.ind_cs_c_data = np.int32(3)

        self.alpha = [np.float64(a) for a in alpha]
        self.alpha_unweighted = [np.float64(1.0) for _ in alpha]
        self.phis_a0 = np.float64(0.0)
        self.ce_0 = self.params["ce0"]
        self.cs_a0 = self.params["cs_a0"]
        self.cs_c0 = self.params["cs_c0"]

        self.ind_deg_i0_a = np.int32(0)
        self.ind_deg_ds_c = np.int32(1)
        self.ind_deg_i0_a_nn = max(self.ind_t, self.ind_r) + self.ind_deg_i0_a
        self.ind_deg_ds_c_nn = max(self.ind_t, self.ind_r) + self.ind_deg_ds_c
        self.dim_params = np.int32(2)
        self.params_min = list(params_min)
        self.params_max = list(params_max)
        self.resc_params = [
            (float(min_val) + float(max_val)) / 2.0
            for (min_val, max_val) in zip(self.params_min, self.params_max)
        ]

        self.hard_IC_timescale = np.float64(hard_IC_timescale)
        self.exponentialLimiter = np.float64(exponentialLimiter)
        self.collocationMode = str(collocationMode).lower()
        self.firstTime = np.float64(firstTime)
        self.tmin_int_bound = np.float64(tmin_int_bound)
        self.dim_inpt = np.int32(2)
        self.nEpochs = int(nEpochs)
        self.nEpochs_lbfgs = int(nEpochs_lbfgs)
        self.nEpochs_start_lbfgs = int(nEpochs_start_lbfgs)
        self.initialLossThreshold = np.float64(initialLossThreshold)
        self.linearizeJ = bool(linearizeJ)
        self.gradualTime_sgd = bool(gradualTime_sgd)
        self.gradualTime_lbfgs = bool(gradualTime_lbfgs)
        self.gradualTimeMode_lbfgs = gradualTimeMode_lbfgs
        self.n_gradual_steps_lbfgs = n_gradual_steps_lbfgs
        self.reg = 0
        self.n_batch = max(int(n_batch), 1)

        self.batch_size_int = int(batch_size_int)
        self.batch_size_bound = int(batch_size_bound)
        self.batch_size_reg = int(batch_size_reg)
        self.max_batch_size_data = int(max_batch_size_data)

        self.activeInt = not (self.batch_size_int == 0 or abs(self.alpha[0]) < 1e-12)
        self.activeBound = not (self.batch_size_bound == 0 or abs(self.alpha[1]) < 1e-12)
        self.activeData = not (self.max_batch_size_data == 0 or abs(self.alpha[2]) < 1e-12 or xDataList == [])
        self.activeReg = not (self.batch_size_reg == 0 or abs(self.alpha[3]) < 1e-12)

        if not self.activeInt:
            self.batch_size_int = 1
        if not self.activeBound:
            self.batch_size_bound = 1
        if not self.activeReg:
            self.batch_size_reg = 1

        self.vprint(f"INFO: Device = {self.device}")
        self.vprint(f"INFO: INT loss is {'ACTIVE' if self.activeInt else 'INACTIVE'}")
        self.vprint(f"INFO: BOUND loss is {'ACTIVE' if self.activeBound else 'INACTIVE'}")
        self.vprint(f"INFO: DATA loss is {'ACTIVE' if self.activeData else 'INACTIVE'}")
        self.vprint(f"INFO: REG loss is {'ACTIVE' if self.activeReg else 'INACTIVE'}")

        if self.activeData:
            for i in range(len(xDataList)):
                checkDataShape(xDataList[i], x_params_dataList[i], yDataList[i])
            ndata = completeDataset(xDataList, x_params_dataList, yDataList)
            self.batch_size_data = min(ndata // self.n_batch, self.max_batch_size_data)
            self.new_nData = self.n_batch * self.batch_size_data
            self.xDataList_full = [np.asarray(x[: self.new_nData], dtype=np.float64) for x in xDataList]
            self.x_params_dataList_full = [np.asarray(x[: self.new_nData], dtype=np.float64) for x in x_params_dataList]
            self.yDataList_full = [np.asarray(y[: self.new_nData], dtype=np.float64) for y in yDataList]
        else:
            self.batch_size_data = 1
            self.new_nData = self.n_batch
            self.xDataList_full = [
                np.zeros((self.n_batch, self.dim_inpt if i in [self.ind_cs_a_data, self.ind_cs_c_data] else self.dim_inpt - 1), dtype=np.float64)
                for i in range(4)
            ]
            self.x_params_dataList_full = [np.zeros((self.n_batch, self.dim_params), dtype=np.float64) for _ in range(4)]
            self.yDataList_full = [np.zeros((self.n_batch, 1), dtype=np.float64) for _ in range(4)]

        self.n_batch_lbfgs = max(int(n_batch_lbfgs), 1)
        if self.n_batch % self.n_batch_lbfgs != 0:
            raise ValueError("n_batch SGD must be divisible by N_BATCH_LBFGS")
        if self.n_batch_lbfgs > self.n_batch:
            raise ValueError("N_BATCH_LBFGS must be smaller or equal to N_BATCH")
        self.batch_size_int_lbfgs = int(self.batch_size_int * self.n_batch / self.n_batch_lbfgs)
        self.batch_size_bound_lbfgs = int(self.batch_size_bound * self.n_batch / self.n_batch_lbfgs)
        self.batch_size_data_lbfgs = int(self.batch_size_data * self.n_batch / self.n_batch_lbfgs)
        self.batch_size_reg_lbfgs = int(self.batch_size_reg * self.n_batch / self.n_batch_lbfgs)

        self.int_col_pts, self.int_col_params = self._build_fixed_interior_collocation(self.batch_size_int * self.n_batch)
        self.bound_col_pts, self.bound_col_params = self._build_fixed_boundary_collocation(self.batch_size_bound * self.n_batch)
        self.reg_col_pts, self.reg_col_params = self._build_fixed_reg_collocation(self.batch_size_reg * self.n_batch)
        self.lbfgs_int_col_pts = None
        self.lbfgs_int_col_params = None
        self.lbfgs_bound_col_pts = None
        self.lbfgs_bound_col_params = None
        self.lbfgs_reg_col_pts = None
        self.lbfgs_reg_col_params = None

        self.model = TorchPINNModel(self).to(self.device)
        self.n_trainable_par = int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.vprint("Num trainable param = ", self.n_trainable_par)

        self.lbfgs = bool(lbfgs and self.nEpochs_lbfgs > 0)
        self.sgd = bool(sgd and self.nEpochs > 0)

        self.config = {
            "hidden_units_t": self.hidden_units_t,
            "hidden_units_t_r": self.hidden_units_t_r,
            "hidden_units_phie": self.hidden_units_phie,
            "hidden_units_phis_c": self.hidden_units_phis_c,
            "hidden_units_cs_a": self.hidden_units_cs_a,
            "hidden_units_cs_c": self.hidden_units_cs_c,
            "n_hidden_res_blocks": self.n_hidden_res_blocks,
            "n_res_block_layers": self.n_res_block_layers,
            "n_res_block_units": self.n_res_block_units,
            "n_grad_path_layers": self.n_grad_path_layers,
            "n_grad_path_units": self.n_grad_path_units,
            "hard_IC_timescale": float(self.hard_IC_timescale),
            "exponentialLimiter": float(self.exponentialLimiter),
            "dynamicAttentionWeights": self.dynamicAttentionWeights,
            "annealingWeights": self.annealingWeights,
            "linearizeJ": self.linearizeJ,
            "activation": self.activation,
            "activeInt": self.activeInt,
            "activeBound": self.activeBound,
            "activeData": self.activeData,
            "activeReg": self.activeReg,
            "params_min": [float(v) for v in self.params_min],
            "params_max": [float(v) for v in self.params_max],
            "local_utilFolder": self.local_utilFolder,
            "hnn_utilFolder": self.hnn_utilFolder,
            "hnn_modelFolder": self.hnn_modelFolder,
            "hnn_params": self.hnn_params,
            "hnntime_utilFolder": self.hnntime_utilFolder,
            "hnntime_modelFolder": self.hnntime_modelFolder,
            "hnntime_val": self.hnntime_val,
        }

        self.best_loss = None
        self.run_SGD = False
        self.run_LBFGS = False
        self.total_step = 0
        self.step = 0

        self.setResidualRescaling(weights)

    def vprint(self, *args):
        if self.verbose:
            print(*args)

    def _rand_uniform(self, shape, minval, maxval) -> torch.Tensor:
        return (
            torch.rand(tuple(shape), dtype=torch.float64, device=self.device)
            * (float(maxval) - float(minval))
            + float(minval)
        )

    def _build_fixed_interior_collocation(self, n_int: int):
        n_int = max(int(n_int), 1)
        r_a_int = self._rand_uniform((n_int, 1), self.rmin + np.float64(1e-12), self.rmax_a)
        r_c_int = self._rand_uniform((n_int, 1), self.rmin + np.float64(1e-12), self.rmax_c)
        r_maxa_int = self.rmax_a * torch.ones((n_int, 1), dtype=torch.float64, device=self.device)
        r_maxc_int = self.rmax_c * torch.ones((n_int, 1), dtype=torch.float64, device=self.device)
        t_max = self.firstTime if self.gradualTime_sgd else self.tmax
        t_int = self._rand_uniform((n_int, 1), self.tmin_int_bound, t_max)
        deg_i0_a_int = self._rand_uniform((n_int, 1), self.params["deg_i0_a_min_eff"], self.params["deg_i0_a_max_eff"])
        deg_ds_c_int = self._rand_uniform((n_int, 1), self.params["deg_ds_c_min_eff"], self.params["deg_ds_c_max_eff"])
        self.ind_int_col_t = np.int32(0)
        self.ind_int_col_r_a = np.int32(1)
        self.ind_int_col_r_c = np.int32(2)
        self.ind_int_col_r_maxa = np.int32(3)
        self.ind_int_col_r_maxc = np.int32(4)
        self.ind_int_col_params_deg_i0_a = np.int32(0)
        self.ind_int_col_params_deg_ds_c = np.int32(1)
        return [t_int, r_a_int, r_c_int, r_maxa_int, r_maxc_int], [deg_i0_a_int, deg_ds_c_int]

    def _build_fixed_boundary_collocation(self, n_bound: int):
        n_bound = max(int(n_bound), 1)
        r_min_bound = torch.zeros((n_bound, 1), dtype=torch.float64, device=self.device)
        r_maxa_bound = self.rmax_a * torch.ones((n_bound, 1), dtype=torch.float64, device=self.device)
        r_maxc_bound = self.rmax_c * torch.ones((n_bound, 1), dtype=torch.float64, device=self.device)
        deg_i0_a_bound = self._rand_uniform((n_bound, 1), self.params["deg_i0_a_min_eff"], self.params["deg_i0_a_max_eff"])
        deg_ds_c_bound = self._rand_uniform((n_bound, 1), self.params["deg_ds_c_min_eff"], self.params["deg_ds_c_max_eff"])
        t_max = self.firstTime if self.gradualTime_sgd else self.tmax
        t_bound = self._rand_uniform((n_bound, 1), self.tmin_int_bound, t_max)
        self.ind_bound_col_t = np.int32(0)
        self.ind_bound_col_r_min = np.int32(1)
        self.ind_bound_col_r_maxa = np.int32(2)
        self.ind_bound_col_r_maxc = np.int32(3)
        self.ind_bound_col_params_deg_i0_a = np.int32(0)
        self.ind_bound_col_params_deg_ds_c = np.int32(1)
        return [t_bound, r_min_bound, r_maxa_bound, r_maxc_bound], [deg_i0_a_bound, deg_ds_c_bound]

    def _build_fixed_reg_collocation(self, n_reg: int):
        n_reg = max(int(n_reg), 1)
        t_max = self.firstTime if self.gradualTime_sgd else self.tmax
        t_reg = self._rand_uniform((n_reg, 1), self.tmin_int_bound, t_max)
        deg_i0_a_reg = self._rand_uniform((n_reg, 1), self.params["deg_i0_a_min_eff"], self.params["deg_i0_a_max_eff"])
        deg_ds_c_reg = self._rand_uniform((n_reg, 1), self.params["deg_ds_c_min_eff"], self.params["deg_ds_c_max_eff"])
        self.ind_reg_col_t = np.int32(0)
        self.ind_reg_col_params_deg_i0_a = np.int32(0)
        self.ind_reg_col_params_deg_ds_c = np.int32(1)
        return [t_reg], [deg_i0_a_reg, deg_ds_c_reg]

    def _freeze_random_collocation_for_lbfgs(self):
        if self.collocationMode != "random":
            self.lbfgs_int_col_pts = self.int_col_pts
            self.lbfgs_int_col_params = self.int_col_params
            self.lbfgs_bound_col_pts = self.bound_col_pts
            self.lbfgs_bound_col_params = self.bound_col_params
            self.lbfgs_reg_col_pts = self.reg_col_pts
            self.lbfgs_reg_col_params = self.reg_col_params
            return
        self.lbfgs_int_col_pts, self.lbfgs_int_col_params = self._build_fixed_interior_collocation(self.batch_size_int_lbfgs * self.n_batch_lbfgs)
        self.lbfgs_bound_col_pts, self.lbfgs_bound_col_params = self._build_fixed_boundary_collocation(self.batch_size_bound_lbfgs * self.n_batch_lbfgs)
        self.lbfgs_reg_col_pts, self.lbfgs_reg_col_params = self._build_fixed_reg_collocation(self.batch_size_reg_lbfgs * self.n_batch_lbfgs)

    def _slice_tensor_batch(self, values, i_batch: int, batch_size: int):
        start = int(i_batch) * int(batch_size)
        end = start + int(batch_size)
        return [value[start:end] for value in values]

    def _slice_np_batch(self, values, i_batch: int, batch_size: int):
        start = int(i_batch) * int(batch_size)
        end = start + int(batch_size)
        return [values_i[start:end] for values_i in values]

    def _to_device_list(self, values):
        return [ensure_2d(v, device=self.device) for v in values]

    def _assemble_batch(self, i_batch: int, use_lbfgs: bool = False):
        if use_lbfgs:
            batch_size_int = self.batch_size_int_lbfgs
            batch_size_bound = self.batch_size_bound_lbfgs
            batch_size_data = self.batch_size_data_lbfgs
            batch_size_reg = self.batch_size_reg_lbfgs
            int_col_pts_full = self.lbfgs_int_col_pts
            int_col_params_full = self.lbfgs_int_col_params
            bound_col_pts_full = self.lbfgs_bound_col_pts
            bound_col_params_full = self.lbfgs_bound_col_params
            reg_col_pts_full = self.lbfgs_reg_col_pts
            reg_col_params_full = self.lbfgs_reg_col_params
        else:
            batch_size_int = self.batch_size_int
            batch_size_bound = self.batch_size_bound
            batch_size_data = self.batch_size_data
            batch_size_reg = self.batch_size_reg
            int_col_pts_full = self.int_col_pts
            int_col_params_full = self.int_col_params
            bound_col_pts_full = self.bound_col_pts
            bound_col_params_full = self.bound_col_params
            reg_col_pts_full = self.reg_col_pts
            reg_col_params_full = self.reg_col_params

        if self.collocationMode == "fixed" or use_lbfgs:
            int_col_pts = self._slice_tensor_batch(int_col_pts_full, i_batch, batch_size_int)
            int_col_params = self._slice_tensor_batch(int_col_params_full, i_batch, batch_size_int)
            bound_col_pts = self._slice_tensor_batch(bound_col_pts_full, i_batch, batch_size_bound)
            bound_col_params = self._slice_tensor_batch(bound_col_params_full, i_batch, batch_size_bound)
            reg_col_pts = self._slice_tensor_batch(reg_col_pts_full, i_batch, batch_size_reg)
            reg_col_params = self._slice_tensor_batch(reg_col_params_full, i_batch, batch_size_reg)
        else:
            int_col_pts = None
            int_col_params = None
            bound_col_pts = None
            bound_col_params = None
            reg_col_pts = None
            reg_col_params = None

        x_batch_trainList = self._slice_np_batch(self.xDataList_full[: self.ind_cs_offset_data], i_batch, batch_size_data)
        x_cs_batch_trainList = self._slice_np_batch(self.xDataList_full[self.ind_cs_offset_data :], i_batch, batch_size_data)
        x_params_batch_trainList = self._slice_np_batch(self.x_params_dataList_full, i_batch, batch_size_data)
        y_batch_trainList = self._slice_np_batch(self.yDataList_full, i_batch, batch_size_data)

        return {
            "int_col_pts": int_col_pts,
            "int_col_params": int_col_params,
            "bound_col_pts": bound_col_pts,
            "bound_col_params": bound_col_params,
            "reg_col_pts": reg_col_pts,
            "reg_col_params": reg_col_params,
            "x_batch_trainList": x_batch_trainList,
            "x_cs_batch_trainList": x_cs_batch_trainList,
            "x_params_batch_trainList": x_params_batch_trainList,
            "y_batch_trainList": y_batch_trainList,
        }

    def _prepare_interior_batch(self, int_col_pts=None, int_col_params=None, tmax=None):
        tmin_int = min(self.tmin_int_bound, self.tmax)
        if self.collocationMode == "random" and int_col_pts is None:
            tmax_eff = self.tmax if tmax is None else float(tmax)
            t = self._rand_uniform((self.batch_size_int, 1), tmin_int, tmax_eff)
            r_a = self._rand_uniform((self.batch_size_int, 1), self.rmin + np.float64(1e-12), self.rmax_a)
            r_c = self._rand_uniform((self.batch_size_int, 1), self.rmin + np.float64(1e-12), self.rmax_c)
            rSurf_a = self.rmax_a * torch.ones((self.batch_size_int, 1), dtype=torch.float64, device=self.device)
            rSurf_c = self.rmax_c * torch.ones((self.batch_size_int, 1), dtype=torch.float64, device=self.device)
            deg_i0_a = self._rand_uniform((self.batch_size_int, 1), self.params["deg_i0_a_min_eff"], self.params["deg_i0_a_max_eff"])
            deg_ds_c = self._rand_uniform((self.batch_size_int, 1), self.params["deg_ds_c_min_eff"], self.params["deg_ds_c_max_eff"])
        else:
            t = ensure_2d(int_col_pts[self.ind_int_col_t], device=self.device)
            if ((self.run_SGD and self.gradualTime_sgd) or (self.run_LBFGS and self.gradualTime_lbfgs)) and tmax is not None:
                t = self.stretchT(t, min(self.tmin_int_bound, self.tmax), self.firstTime, min(self.tmin_int_bound, self.tmax), float(tmax))
            r_a = ensure_2d(int_col_pts[self.ind_int_col_r_a], device=self.device)
            r_c = ensure_2d(int_col_pts[self.ind_int_col_r_c], device=self.device)
            rSurf_a = ensure_2d(int_col_pts[self.ind_int_col_r_maxa], device=self.device)
            rSurf_c = ensure_2d(int_col_pts[self.ind_int_col_r_maxc], device=self.device)
            deg_i0_a = ensure_2d(int_col_params[self.ind_int_col_params_deg_i0_a], device=self.device)
            deg_ds_c = ensure_2d(int_col_params[self.ind_int_col_params_deg_ds_c], device=self.device)
        return t, r_a, r_c, rSurf_a, rSurf_c, deg_i0_a, deg_ds_c

    def _prepare_boundary_batch(self, bound_col_pts=None, bound_col_params=None, tmax=None):
        tmin_bound = min(self.tmin_int_bound, self.tmax)
        if self.collocationMode == "random" and bound_col_pts is None:
            tmax_eff = self.tmax if tmax is None else float(tmax)
            t_bound = self._rand_uniform((self.batch_size_bound, 1), tmin_bound, tmax_eff)
            r_0_bound = torch.zeros((self.batch_size_bound, 1), dtype=torch.float64, device=self.device)
            r_max_a_bound = self.rmax_a * torch.ones((self.batch_size_bound, 1), dtype=torch.float64, device=self.device)
            r_max_c_bound = self.rmax_c * torch.ones((self.batch_size_bound, 1), dtype=torch.float64, device=self.device)
            deg_i0_a_bound = self._rand_uniform((self.batch_size_bound, 1), self.params["deg_i0_a_min_eff"], self.params["deg_i0_a_max_eff"])
            deg_ds_c_bound = self._rand_uniform((self.batch_size_bound, 1), self.params["deg_ds_c_min_eff"], self.params["deg_ds_c_max_eff"])
        else:
            t_bound = ensure_2d(bound_col_pts[self.ind_bound_col_t], device=self.device)
            if ((self.run_SGD and self.gradualTime_sgd) or (self.run_LBFGS and self.gradualTime_lbfgs)) and tmax is not None:
                t_bound = self.stretchT(t_bound, min(self.tmin_int_bound, self.tmax), self.firstTime, min(self.tmin_int_bound, self.tmax), float(tmax))
            r_0_bound = ensure_2d(bound_col_pts[self.ind_bound_col_r_min], device=self.device)
            r_max_a_bound = ensure_2d(bound_col_pts[self.ind_bound_col_r_maxa], device=self.device)
            r_max_c_bound = ensure_2d(bound_col_pts[self.ind_bound_col_r_maxc], device=self.device)
            deg_i0_a_bound = ensure_2d(bound_col_params[self.ind_bound_col_params_deg_i0_a], device=self.device)
            deg_ds_c_bound = ensure_2d(bound_col_params[self.ind_bound_col_params_deg_ds_c], device=self.device)
        return t_bound, r_0_bound, r_max_a_bound, r_max_c_bound, deg_i0_a_bound, deg_ds_c_bound

    def stretchT(self, t, tmin, tmax, tminp, tmaxp):
        t = ensure_2d(t, device=self.device)
        return (t - float(tmin)) * (float(tmaxp) - float(tminp)) / (float(tmax) - float(tmin) + 1e-16) + float(tminp)

    def _prepare_scalars_for_json(self, data):
        if isinstance(data, dict):
            return {k: self._prepare_scalars_for_json(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._prepare_scalars_for_json(v) for v in data]
        if isinstance(data, np.generic):
            return data.item()
        return data

    def prepareLog(self):
        os.makedirs(self.modelFolder, exist_ok=True)
        os.makedirs(self.logLossFolder, exist_ok=True)

        config_path = os.path.join(self.modelFolder, "config.json")
        with open(config_path, "w", encoding="utf-8") as outfile:
            json.dump(self._prepare_scalars_for_json(self.config), outfile, indent=4, sort_keys=True)

        header_files = {
            "log.csv": "epoch;step;mseloss\n",
            "interiorTerms.csv": "step;lossArray\n",
            "boundaryTerms.csv": "step;lossArray\n",
            "dataTerms.csv": "step;lossArray\n",
            "regTerms.csv": "step;lossArray\n",
        }
        for filename, header in header_files.items():
            with open(os.path.join(self.logLossFolder, filename), "w", encoding="utf-8") as f:
                f.write(header)

    def initLearningRateControl(self, learningRateModel, learningRateWeights=None):
        with open(os.path.join(self.modelFolder, "learningRateModel"), "w", encoding="utf-8") as f:
            f.write(str(learningRateModel))
        if learningRateWeights is not None:
            with open(os.path.join(self.modelFolder, "learningRateWeights"), "w", encoding="utf-8") as f:
                f.write(str(learningRateWeights))

    def initLossThresholdControl(self, lossThreshold):
        with open(os.path.join(self.modelFolder, "lossThreshold"), "w", encoding="utf-8") as f:
            f.write(str(lossThreshold))

    def _append_log_line(self, filename: str, line: str):
        with open(os.path.join(self.logLossFolder, filename), "a", encoding="utf-8") as f:
            f.write(line)

    def _terms_to_line(self, step: int, terms):
        vals = [float(torch.mean(torch.square(term[0])).detach().cpu()) for term in terms]
        return f"{step};" + " ".join(f"{v:.12e}" for v in vals) + "\n"

    def _compute_loss_from_batch(self, batch, tmax=None, weighted=True):
        interiorTerms = self.interior_loss(batch["int_col_pts"], batch["int_col_params"], tmax)
        boundaryTerms = self.boundary_loss(batch["bound_col_pts"], batch["bound_col_params"], tmax)
        dataTerms = self.data_loss(
            batch["x_batch_trainList"],
            batch["x_cs_batch_trainList"],
            batch["x_params_batch_trainList"],
            batch["y_batch_trainList"],
        )
        regTerms = self.regularization_loss(batch["reg_col_pts"], tmax)

        if weighted:
            interiorTerms_rescaled = [term[0] * float(resc) for term, resc in zip(interiorTerms, self.interiorTerms_rescale)]
            boundaryTerms_rescaled = [term[0] * float(resc) for term, resc in zip(boundaryTerms, self.boundaryTerms_rescale)]
            dataTerms_rescaled = [term[0] * float(resc) for term, resc in zip(dataTerms, self.dataTerms_rescale)]
            regTerms_rescaled = [term[0] * float(resc) for term, resc in zip(regTerms, self.regTerms_rescale)]
            loss_value, int_loss, bound_loss, data_loss, reg_loss = loss_fn(
                interiorTerms_rescaled,
                boundaryTerms_rescaled,
                dataTerms_rescaled,
                regTerms_rescaled,
                alpha=self.alpha,
            )
        else:
            interiorTerms_rescaled = [term[0] * float(resc) for term, resc in zip(interiorTerms, self.interiorTerms_rescale_unweighted)]
            boundaryTerms_rescaled = [term[0] * float(resc) for term, resc in zip(boundaryTerms, self.boundaryTerms_rescale_unweighted)]
            dataTerms_rescaled = [term[0] * float(resc) for term, resc in zip(dataTerms, self.dataTerms_rescale_unweighted)]
            regTerms_rescaled = [term[0] * float(resc) for term, resc in zip(regTerms, self.regTerms_rescale_unweighted)]
            loss_value, int_loss, bound_loss, data_loss, reg_loss = loss_fn(
                interiorTerms_rescaled,
                boundaryTerms_rescaled,
                dataTerms_rescaled,
                regTerms_rescaled,
                alpha=self.alpha_unweighted,
            )

        return {
            "loss": loss_value,
            "int_loss": int_loss,
            "bound_loss": bound_loss,
            "data_loss": data_loss,
            "reg_loss": reg_loss,
            "interiorTerms": interiorTerms,
            "boundaryTerms": boundaryTerms,
            "dataTerms": dataTerms,
            "regTerms": regTerms,
            "interiorTerms_rescaled": interiorTerms_rescaled,
            "boundaryTerms_rescaled": boundaryTerms_rescaled,
            "dataTerms_rescaled": dataTerms_rescaled,
            "regTerms_rescaled": regTerms_rescaled,
        }

    def train_step(
        self,
        int_col_pts=None,
        int_col_params=None,
        bound_col_pts=None,
        bound_col_params=None,
        reg_col_pts=None,
        reg_col_params=None,
        x_batch_trainList=None,
        x_cs_batch_trainList=None,
        x_params_batch_trainList=None,
        y_batch_trainList=None,
        tmax=None,
        gradient_threshold=None,
    ):
        self.model.train(True)
        self.model.zero_grad(set_to_none=True)
        batch = {
            "int_col_pts": int_col_pts,
            "int_col_params": int_col_params,
            "bound_col_pts": bound_col_pts,
            "bound_col_params": bound_col_params,
            "reg_col_pts": reg_col_pts,
            "reg_col_params": reg_col_params,
            "x_batch_trainList": x_batch_trainList,
            "x_cs_batch_trainList": x_cs_batch_trainList,
            "x_params_batch_trainList": x_params_batch_trainList,
            "y_batch_trainList": y_batch_trainList,
        }
        out = self._compute_loss_from_batch(batch, tmax=tmax, weighted=True)
        out["loss"].backward()
        if gradient_threshold is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_threshold))
        grads_model = [None if p.grad is None else p.grad.detach().clone() for p in self.model.parameters()]
        denom = float(out["loss"].detach().cpu()) + 1e-16
        return (
            out["loss"],
            out["int_loss"] / denom,
            out["bound_loss"] / denom,
            out["data_loss"] / denom,
            out["reg_loss"] / denom,
            out["interiorTerms_rescaled"],
            out["boundaryTerms_rescaled"],
            out["dataTerms_rescaled"],
            out["regTerms_rescaled"],
            grads_model,
        )

    def _save_best_and_last(self, loss_value: float):
        last_h5 = os.path.join(self.modelFolder, "last.weights.h5")
        best_h5 = os.path.join(self.modelFolder, "best.weights.h5")
        last_pt = os.path.join(self.modelFolder, "last.pt")
        best_pt = os.path.join(self.modelFolder, "best.pt")
        safe_save(self.model, last_h5, overwrite=True)
        safe_save(self.model, last_pt, overwrite=True)
        if self.best_loss is None or loss_value < self.best_loss:
            self.best_loss = float(loss_value)
            safe_save(self.model, best_h5, overwrite=True)
            safe_save(self.model, best_pt, overwrite=True)

    def _set_optimizer_lr(self, optimizer: torch.optim.Optimizer, lr_value: float):
        for group in optimizer.param_groups:
            group["lr"] = float(lr_value)

    def _compute_unweighted_dataset_loss(self, use_lbfgs: bool = False, tmax=None):
        n_batches = self.n_batch_lbfgs if use_lbfgs else self.n_batch
        losses = []
        with torch.enable_grad():
            for i_batch in range(int(n_batches)):
                batch = self._assemble_batch(i_batch, use_lbfgs=use_lbfgs)
                out = self._compute_loss_from_batch(batch, tmax=tmax, weighted=False)
                losses.append(out["loss"])
        return float(torch.mean(torch.stack(losses)).detach().cpu())

    def _lbfgs_epoch(self, optimizer, epoch_idx: int, tmax=None, gradient_threshold=None):
        self.model.train(True)
        epoch_losses = []

        def closure():
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), dtype=torch.float64, device=self.device)
            for i_batch in range(self.n_batch_lbfgs):
                batch = self._assemble_batch(i_batch, use_lbfgs=True)
                out = self._compute_loss_from_batch(batch, tmax=tmax, weighted=True)
                total_loss = total_loss + out["loss"] / float(self.n_batch_lbfgs)
            total_loss.backward()
            if gradient_threshold is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_threshold))
            epoch_losses.append(float(total_loss.detach().cpu()))
            return total_loss

        loss_tensor = optimizer.step(closure)
        loss_value = float(loss_tensor.detach().cpu()) if isinstance(loss_tensor, torch.Tensor) else float(epoch_losses[-1])
        self._save_best_and_last(loss_value)
        self._append_log_line("log.csv", f"{epoch_idx};0;{loss_value:.12e}\n")
        return loss_value

    def train(
        self,
        learningRateModel,
        learningRateModelFinal,
        lrSchedulerModel,
        learningRateWeights=None,
        learningRateWeightsFinal=None,
        lrSchedulerWeights=None,
        learningRateLBFGS=None,
        inner_epochs=None,
        start_weight_training_epoch=None,
        gradient_threshold=None,
    ):
        if gradient_threshold is not None:
            print(f"INFO: clipping gradients at {gradient_threshold:.2g}")

        self.prepareLog()
        self.initLearningRateControl(learningRateModel, learningRateWeights)
        self.initLossThresholdControl(self.initialLossThreshold)

        self.run_SGD = False
        self.run_LBFGS = False
        lr_m = float(learningRateModel)

        total_epochs_done = 0
        if self.sgd:
            print(f"Using collocation points: {self.collocationMode}")
            adam = torch.optim.Adam(self.model.parameters(), lr=lr_m)
            self.run_SGD = True
            for epoch in range(self.nEpochs):
                if lrSchedulerModel is not None:
                    lr_m = float(lrSchedulerModel(epoch, lr_m))
                    self._set_optimizer_lr(adam, lr_m)
                epoch_loss = 0.0
                last_batch_terms = None
                for i_batch in range(self.n_batch):
                    self.total_step = i_batch + total_epochs_done * self.n_batch
                    self.step = i_batch
                    batch = self._assemble_batch(i_batch, use_lbfgs=False)
                    adam.zero_grad(set_to_none=True)
                    out = self._compute_loss_from_batch(batch, tmax=None, weighted=True)
                    out["loss"].backward()
                    if gradient_threshold is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_threshold))
                    adam.step()
                    loss_value = float(out["loss"].detach().cpu())
                    epoch_loss += loss_value / float(self.n_batch)
                    last_batch_terms = out
                    self._append_log_line("log.csv", f"{epoch};{i_batch};{loss_value:.12e}\n")
                if last_batch_terms is not None:
                    self._append_log_line("interiorTerms.csv", self._terms_to_line(epoch, last_batch_terms["interiorTerms"]))
                    self._append_log_line("boundaryTerms.csv", self._terms_to_line(epoch, last_batch_terms["boundaryTerms"]))
                    self._append_log_line("dataTerms.csv", self._terms_to_line(epoch, last_batch_terms["dataTerms"]))
                    self._append_log_line("regTerms.csv", self._terms_to_line(epoch, last_batch_terms["regTerms"]))
                self._save_best_and_last(epoch_loss)
                total_epochs_done += 1

        if self.lbfgs:
            self._freeze_random_collocation_for_lbfgs()
            self.run_SGD = False
            self.run_LBFGS = True
            optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=float(learningRateLBFGS),
                max_iter=1,
                history_size=100,
                line_search_fn="strong_wolfe",
            )
            for epoch_lbfgs in range(self.nEpochs_lbfgs):
                loss_value = self._lbfgs_epoch(
                    optimizer,
                    total_epochs_done + epoch_lbfgs,
                    tmax=None,
                    gradient_threshold=gradient_threshold,
                )
            total_epochs_done += self.nEpochs_lbfgs

        if os.path.isfile(os.path.join(self.modelFolder, "best.weights.h5")):
            payload = torch.load(os.path.join(self.modelFolder, "best.weights.h5"), map_location=self.device)
            if isinstance(payload, dict) and "model_state_dict" in payload:
                self.model.load_state_dict(payload["model_state_dict"])
            elif isinstance(payload, dict):
                self.model.load_state_dict(payload)
        return self._compute_unweighted_dataset_loss(use_lbfgs=self.lbfgs)

    def runLBFGS(
        self,
        tmax,
        nIter,
        epochDoneLBFGS,
        epochDoneSGD,
        bestLoss,
        learningRateLBFGS,
        gradient_threshold=None,
    ):
        self._freeze_random_collocation_for_lbfgs()
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=float(learningRateLBFGS),
            max_iter=1,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        for i in range(int(nIter)):
            loss_value = self._lbfgs_epoch(optimizer, int(epochDoneSGD) + int(epochDoneLBFGS) + i, tmax=tmax, gradient_threshold=gradient_threshold)
            if bestLoss is None:
                bestLoss = loss_value
            else:
                bestLoss = min(float(bestLoss), float(loss_value))
        return bestLoss


from _rescale import (
    fix_param,
    get_cs_a_hnn,
    get_cs_a_hnntime,
    get_cs_c_hnn,
    get_cs_c_hnntime,
    get_phie0,
    get_phie_hnn,
    get_phie_hnntime,
    get_phis_c0,
    get_phis_c_hnn,
    get_phis_c_hnntime,
    rescale_param,
    rescaleCs_a,
    rescaleCs_c,
    rescalePhie,
    rescalePhis_c,
    unrescale_param,
)
from _losses import (
    boundary_loss,
    data_loss,
    get_loss_and_flat_grad,
    loss_fn,
    get_loss_and_flat_grad_SA,
    get_loss_and_flat_grad_annealing,
    get_unweighted_loss,
    interior_loss,
    regularization_loss,
    setResidualRescaling,
)


# Bind helper functions as class methods for compatibility with the original code path.
myNN.fix_param = fix_param
myNN.get_cs_a_hnn = get_cs_a_hnn
myNN.get_cs_a_hnntime = get_cs_a_hnntime
myNN.get_cs_c_hnn = get_cs_c_hnn
myNN.get_cs_c_hnntime = get_cs_c_hnntime
myNN.get_phie0 = get_phie0
myNN.get_phie_hnn = get_phie_hnn
myNN.get_phie_hnntime = get_phie_hnntime
myNN.get_phis_c0 = get_phis_c0
myNN.get_phis_c_hnn = get_phis_c_hnn
myNN.get_phis_c_hnntime = get_phis_c_hnntime
myNN.rescale_param = rescale_param
myNN.rescaleCs_a = rescaleCs_a
myNN.rescaleCs_c = rescaleCs_c
myNN.rescalePhie = rescalePhie
myNN.rescalePhis_c = rescalePhis_c
myNN.unrescale_param = unrescale_param
myNN.boundary_loss = boundary_loss
myNN.data_loss = data_loss
myNN.get_loss_and_flat_grad = get_loss_and_flat_grad
myNN.get_loss_and_flat_grad_SA = get_loss_and_flat_grad_SA
myNN.get_loss_and_flat_grad_annealing = get_loss_and_flat_grad_annealing
myNN.get_unweighted_loss = get_unweighted_loss
myNN.interior_loss = interior_loss
myNN.regularization_loss = regularization_loss
myNN.setResidualRescaling = setResidualRescaling
