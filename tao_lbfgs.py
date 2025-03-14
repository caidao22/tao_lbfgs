import petsc4py
from petsc4py import PETSc
from unittest.mock import patch
try:
    import torch
    from torch.optim import Optimizer
except ImportError:
    torch = None
    Optimizer = object

@patch.dict("sys.modules", torch=torch)
class TAOtorch(torch.optim.Optimizer):
    r"""PyTorch.Optimizer() wrapper for TAO solvers.

    This class makes TAO solvers mimic traditional PyTorch.Optimizer() objects
    by performing single-iteration tao.solve() calls for each optimizer.step()
    in a training cycle.

    This implementation incorporates an adaptive learning rate based on
    `RMSprop <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_
    and `AMSgrad <https://openreview.net/forum?id=ryQu7f-RZ>`_. The learning rate
    is baked into the TAO gradient evaluation, and the TAO line search is set to accept
    unit step length.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        tao (PETSc.TAO, optional): PETSc.TAO solver object (default: TAOBNCG)
        adaptive (str, optional): choose between 'amsgrad, 'rmsgrad', and None (default: 'amsgrad')
        rhp (float, optional): RMS decay rate (default: 0.9)
        eps (float, optional): RMS zero safeguard (default: 1e-8)
    """

    def __init__(self, params, lr=1.0, adaptive='amsgrad', rho=0.9, eps=1e-8):
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid step direction RMS decay rate: {}".format(rho))
        if not 0.0 < eps:
            raise ValueError("Invalid zero safeguard: {}".format(eps))
        if adaptive not in ['amsgrad', 'rmsgrad', None]:
            raise ValueError("Invalid adaptive LR method: {}".format(adaptive))
        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            adaptive=adaptive,
            dir_sq_avg=None,
            dir_sq_avg_max=None)
        super(TAOtorch, self).__init__(params, defaults)
        # these two vectors below share memory!!
        self.flatpar = None   # torch "flat" parameter tensor
        self.paramvec = None  # TAO solution vector
        # intialize the TAO solver
        self.tao = None
        self.tao = self.getTAO()

    def _getParams(self, zero=False, grad=False):
        flatpar = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    flatpar.append(p.detach().clone().view(-1))
                    if zero:
                        flatpar[-1][:] = 0.0
                    elif grad:
                        if p.grad is not None:
                            if p.grad.is_sparse:
                                flatpar[-1][:] = p.grad.to_dense().view(-1)[:]
                            else:
                                flatpar[-1][:] = p.grad.view(-1)[:]
                        else:
                            flatpar[-1][:] = 0.0
        return torch.cat(flatpar, 0)

    def _setParams(self, flatpar):
        begin = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    end = begin + len(p.view(-1))
                    p.copy_(torch.reshape(flatpar[begin:end], p.shape))
                    begin = end

    def _computeAdaptiveLR(self, ds):
        lr_type = self.defaults['adaptive']
        if lr_type in ['amsgrad', 'rmsgrad']:
            rho = self.defaults['rho']
            eps = self.defaults['eps']
            dir_sq_avg = self.defaults['dir_sq_avg']
            dir_sq_avg_max = self.defaults['dir_sq_avg_max']
            if dir_sq_avg is None:
                # this is the first step so create data structures to track average mean squares
                dir_sq_avg = torch.zeros_like(ds)
                dir_sq_avg.addcmul_(ds, ds, value=1.-rho)
                if lr_type == 'amsgrad':
                    dir_sq_avg_max = dir_sq_avg.detach().clone()
                else:
                    dir_sq_avg_max = dir_sq_avg
            else:
                dir_sq_avg.mul_(rho).addcmul_(ds, ds, value=1.-rho)
                if lr_type == 'amsgrad':
                    torch.maximum(dir_sq_avg_max, dir_sq_avg, out=dir_sq_avg_max)
                else:
                    dir_sq_avg_max = dir_sq_avg
            return self.defaults['lr']/dir_sq_avg_max.add(eps).sqrt_()
        else:
            return self.defaults['lr']

    def _evalObjGrad(self, tao, x, G):
        # assume that loss.backward() has already been called before tao.step()
        # create a flattened parameter tensor that shares memory with the TAO gradient vector
        G.attachDLPackInfo(vec=self.paramvec)
        flatgrad = torch.utils.dlpack.from_dlpack(G)
        # copy NN gradients into the flattened tensor
        flatgrad.copy_(self._getParams(grad=True))
        # scale the gradient with RMSgrad/AMSGrad adaptive learning rate
        lr = self._computeAdaptiveLR(flatgrad)
        flatgrad.mul_(lr)
        return flatgrad.norm(2)

    def _configureTAO(self, tao):
        if self.paramvec is not None:
            tao.setInitial(self.paramvec)
            if self.flatpar is None:
                self.flatpar = torch.utils.dlpack.from_dlpack(self.paramvec)
        else:
            self.flatpar = self._getParams()
            self.paramvec = PETSc.Vec().createWithDLPack(self.flatpar)
            tao.setInitial(self.paramvec)
        tao.setObjectiveGradient(self._evalObjGrad)
        tao.setMaximumIterations(1)
        tao.setTolerances(gatol=0.0, gttol=0.0, grtol=0.0)
        # ls = tao.getLineSearch()
        # ls.setType('unit')
        return tao

    def getTAO(self):
        if self.tao is None:
            tao = PETSc.TAO().create(comm=PETSc.COMM_SELF)
            tao.setOptionsPrefix('torch_')
            tao.setType('bqnktr')
            tao.setFromOptions()
            tao = self._configureTAO(tao)
            self.tao = tao
        return self.tao

    def setTAO(self, tao):
        assert isinstance(tao, PETSc.TAO)
        tao = self._configureTAO(tao)
        self.tao.destroy()
        self.tao = tao

    def step(self, closure=None):
        # get the NN parameters and write into the flattened tensor
        self.flatpar.copy_(self._getParams())
        # trigger the tao solution (for 1 iteration)
        self.tao.solve()
        # write the updated solution to NN parameters
        with torch.no_grad():
            self._setParams(self.flatpar)

    def destroy(self):
        self.tao.destroy()
        self.paramvec.destroy()

