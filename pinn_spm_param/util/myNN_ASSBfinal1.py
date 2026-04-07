from __future__ import annotations

from myNN import myNN as _BaseMyNN

from _losses_ASSBfinal1 import (
    boundary_loss,
    data_loss,
    get_loss_and_flat_grad,
    get_loss_and_flat_grad_SA,
    get_loss_and_flat_grad_annealing,
    get_unweighted_loss,
    interior_loss,
    loss_fn,
    regularization_loss,
    setResidualRescaling,
)
from _rescale_ASSBfinal1 import (
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


myNN = _BaseMyNN

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
