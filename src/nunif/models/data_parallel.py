import torch
import torch.nn as nn
from torch.nn.parallel import gather, replicate, parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs
from .. device import create_device


class DataParallelInference():
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        self.dim = dim
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = self.device_ids[0]
        self.output_device = output_device
        self.module = module.eval()
        self.replicas = replicate(self.module, self.device_ids)

    def __call__(self, inputs, **kwargs):
        if not isinstance(inputs, tuple):
            inputs = (inputs,) if inputs is not None else ()
        inputs, module_kwargs = scatter_kwargs(inputs, kwargs, self.device_ids, self.dim)
        if not inputs and not module_kwargs:
            inputs = ((),)
            module_kwargs = ({},)

        # if len(self.device_ids) == 1:
        #    return self.module(*inputs[0], **module_kwargs[0])

        used_device_ids = self.device_ids[:len(inputs)]
        replicas = self.replicas[:len(inputs)]
        outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
        return gather(outputs, self.output_device, self.dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DataParallelWrapper(nn.DataParallel):
    # ref: https://discuss.pytorch.org/t/making-a-wrapper-around-nn-dataparallel-to-access-module-attributes-is-safe/79124
    def __init__(self, module, device_ids=None):
        super().__init__(module, device_ids=device_ids)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DeviceSwitchInference(object):
    def __init__(self, module, device_ids=None):
        device_ids = device_ids or list(range(torch.cuda.device_count()))
        self._devices = [create_device(id) for id in device_ids]
        self._module = module.eval()
        self._replicas = replicate(self._module, self._devices)

    def __call__(self, x, *args, **kwargs):
        module = self._replicas[self._devices.index(x.device)]
        return module(x, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)
