from __future__ import annotations

import numpy as np
import torch

from torch_utils import ensure_2d, to_tensor


torch.set_default_dtype(torch.float64)


def _reshape_like_ASSBfinal1(value, ref):
    tensor = ensure_2d(value, device=ref.device)
    return tensor.reshape_as(ref)


def rescalePhie(self, phie, t, deg_i0_a, deg_ds_c):
    del deg_ds_c
    phie = ensure_2d(phie, device=self.device)
    t_reshape = _reshape_like_ASSBfinal1(t, phie)
    deg_i0_a_reshape = _reshape_like_ASSBfinal1(deg_i0_a, phie)
    if self.use_hnntime:
        raise NotImplementedError("HNNTIME is not ported in the PyTorch patch.")
    phie_start = self.get_phie0(deg_i0_a_reshape)
    time_distance = np.float64(1.0) - torch.exp(-(t_reshape) / self.hard_IC_timescale)
    if self.use_hnn:
        raise NotImplementedError("HNN is not ported in the PyTorch patch.")
    excursion = to_tensor(self.params["phie_excursion_est"], device=self.device)
    phie_nn = torch.tanh(phie)
    return phie_start + time_distance * excursion * phie_nn


def rescalePhis_c(self, phis_c, t, deg_i0_a, deg_ds_c):
    del deg_ds_c
    phis_c = ensure_2d(phis_c, device=self.device)
    t_reshape = _reshape_like_ASSBfinal1(t, phis_c)
    deg_i0_a_reshape = _reshape_like_ASSBfinal1(deg_i0_a, phis_c)
    if self.use_hnntime:
        raise NotImplementedError("HNNTIME is not ported in the PyTorch patch.")
    phis_c_start = self.get_phis_c0(deg_i0_a_reshape)
    time_distance = np.float64(1.0) - torch.exp(-(t_reshape) / self.hard_IC_timescale)
    if self.use_hnn:
        raise NotImplementedError("HNN is not ported in the PyTorch patch.")
    excursion = to_tensor(self.params["phis_c_excursion_est"], device=self.device)
    phis_c_nn = torch.tanh(phis_c)
    return phis_c_start + time_distance * excursion * phis_c_nn


def rescaleCs_a(self, cs_a, t, r, deg_i0_a, deg_ds_c, clip=True):
    del r, deg_i0_a, deg_ds_c
    cs_a = ensure_2d(cs_a, device=self.device)
    t_reshape = _reshape_like_ASSBfinal1(t, cs_a)
    if self.use_hnntime:
        raise NotImplementedError("HNNTIME is not ported in the PyTorch patch.")
    cs_a_start = to_tensor(self.cs_a0, device=self.device) + torch.zeros_like(cs_a)
    time_distance = np.float64(1.0) - torch.exp(-(t_reshape) / self.hard_IC_timescale)
    if self.use_hnn:
        raise NotImplementedError("HNN is not ported in the PyTorch patch.")
    excursion = to_tensor(self.params["cs_a_excursion_est"], device=self.device)
    out = cs_a_start + time_distance * excursion * torch.tanh(cs_a)
    if clip:
        out = torch.clamp(out, np.float64(0.0), float(self.params["csanmax"]))
    return out


def rescaleCs_c(self, cs_c, t, r, deg_i0_a, deg_ds_c, clip=True):
    del r, deg_i0_a, deg_ds_c
    cs_c = ensure_2d(cs_c, device=self.device)
    t_reshape = _reshape_like_ASSBfinal1(t, cs_c)
    if self.use_hnntime:
        raise NotImplementedError("HNNTIME is not ported in the PyTorch patch.")
    cs_c_start = to_tensor(self.cs_c0, device=self.device) + torch.zeros_like(cs_c)
    time_distance = np.float64(1.0) - torch.exp(-(t_reshape) / self.hard_IC_timescale)
    if self.use_hnn:
        raise NotImplementedError("HNN is not ported in the PyTorch patch.")
    excursion = to_tensor(self.params["cs_c_excursion_est"], device=self.device)
    out = cs_c_start + time_distance * excursion * torch.tanh(cs_c)
    if clip:
        out = torch.clamp(out, np.float64(0.0), float(self.params["cscamax"]))
    return out


def get_phie0(self, deg_i0_a):
    deg_i0_a = ensure_2d(deg_i0_a, device=self.device)
    i0_a = self.params["i0_a"](
        self.params["cs_a0"] * torch.ones_like(deg_i0_a),
        self.params["ce0"] * torch.ones_like(deg_i0_a),
        self.params["T"],
        self.params["alpha_a"],
        self.params["csanmax"],
        self.params["R"],
        deg_i0_a,
    )
    return self.params["phie0"](
        i0_a,
        self.params["j_a"],
        self.params["F"],
        self.params["R"],
        self.params["T"],
        self.params["Uocp_a0"],
    )


def get_phis_c0(self, deg_i0_a):
    deg_i0_a = ensure_2d(deg_i0_a, device=self.device)
    i0_a = self.params["i0_a"](
        self.params["cs_a0"] * torch.ones_like(deg_i0_a),
        self.params["ce0"] * torch.ones_like(deg_i0_a),
        self.params["T"],
        self.params["alpha_a"],
        self.params["csanmax"],
        self.params["R"],
        deg_i0_a,
    )
    return self.params["phis_c0"](
        i0_a,
        self.params["j_a"],
        self.params["F"],
        self.params["R"],
        self.params["T"],
        self.params["Uocp_a0"],
        self.params["j_c"],
        self.params["i0_c0"],
        self.params["Uocp_c0"],
    )


def get_phie_hnn(self, t, deg_i0_a, deg_ds_c):
    raise NotImplementedError("HNN is not ported in the PyTorch patch.")


def get_phie_hnntime(self, deg_i0_a, deg_ds_c):
    raise NotImplementedError("HNNTIME is not ported in the PyTorch patch.")


def get_phis_c_hnn(self, t, deg_i0_a, deg_ds_c):
    raise NotImplementedError("HNN is not ported in the PyTorch patch.")


def get_phis_c_hnntime(self, deg_i0_a, deg_ds_c):
    raise NotImplementedError("HNNTIME is not ported in the PyTorch patch.")


def get_cs_a_hnn(self, t, r, deg_i0_a, deg_ds_c):
    raise NotImplementedError("HNN is not ported in the PyTorch patch.")


def get_cs_a_hnntime(self, r, deg_i0_a, deg_ds_c):
    raise NotImplementedError("HNNTIME is not ported in the PyTorch patch.")


def get_cs_c_hnn(self, t, r, deg_i0_a, deg_ds_c):
    raise NotImplementedError("HNN is not ported in the PyTorch patch.")


def get_cs_c_hnntime(self, r, deg_i0_a, deg_ds_c):
    raise NotImplementedError("HNNTIME is not ported in the PyTorch patch.")


def rescale_param(self, param, ind):
    param_t = ensure_2d(param, device=self.device)
    return (param_t - float(self.params_min[ind])) / float(self.resc_params[ind])


def fix_param(self, param, param_val):
    param_t = ensure_2d(param, device=self.device)
    return float(param_val) * torch.ones_like(param_t, dtype=torch.float64)


def unrescale_param(self, param_rescaled, ind):
    param_rescaled = ensure_2d(param_rescaled, device=self.device)
    return param_rescaled * float(self.resc_params[ind]) + float(self.params_min[ind])
