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

    def __init__(self, params, closure=None, lr=1.0, max_iter=20, max_eval=None, tolerance_grad: float=1e-7, tolerance_change: float=1e-9, history_size: int=100):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
        )
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

    def _evalObjGrad(self, tao, x, G):
        loss = float(self.closure())
        # create a flattened parameter tensor that shares memory with the TAO gradient vector
        G.attachDLPackInfo(vec=self.paramvec)
        flatgrad = torch.utils.dlpack.from_dlpack(G)
        # copy NN gradients into the flattened tensor
        flatgrad.copy_(self._getParams(grad=True))
        # scale the gradient with fixed learning rate
        flatgrad.mul_(self.defaults['lr'])
        return loss

    def _configureTAO(self, tao):
        if self.paramvec is not None:
            tao.setInitial(self.paramvec)
            if self.flatpar is None:
                self.flatpar = torch.utils.dlpack.from_dlpack(self.paramvec)
        else:
            self.flatpar = self._getParams()
            self.paramvec = PETSc.Vec().createWithDLPack(self.flatpar)
            tao.setInitial(self.paramvec)
        tao.setMaximumIterations(self.defaults["max_iter"])
        tao.setTolerances(gatol=self.defaults["tolerance_grad"], grtol=0.0, gttol=self.defaults["tolerance_change"])
        # ls = tao.getLineSearch()
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

    def step(self, closure):
        # Make sure the closure is always called with grad enabled
        self.closure = torch.enable_grad()(closure)
        self.tao.setObjectiveGradient(self._evalObjGrad)
        # get the NN parameters and write into the flattened tensor
        self.flatpar.copy_(self._getParams())
        # trigger the tao solution (for 1 iteration)
        self.tao.solve()
        # write the updated solution to NN parameters
        with torch.no_grad():
            self._setParams(self.flatpar)
        return torch.tensor([self.tao.getObjectiveValue()])

    def destroy(self):
        self.tao.destroy()
        self.paramvec.destroy()

