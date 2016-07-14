import os
import numpy as np
import scipy.io
from datetime import datetime
import theano.tensor as T
import utils
import time
import logging


logger = logging.getLogger()


class Experiment(object):
    """
    Experiment. For easy saving and loading.
    """
    def __init__(self, name, dump_path, create_logger=True):
        """
        Initialize the experiment.
        """
        self.name = name
        self.start_time = datetime.now()
        self.dump_path = os.path.join(dump_path, name)
        self.components = {}
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
        self.logs_path = os.path.join(self.dump_path, "experiment.log")
        if create_logger:
            self.log_formatter = utils.create_logger(self.logs_path)

    def reset_time(self):
        """
        Reset start time (for logs).
        """
        self.log_formatter.start_time = time.time()

    def add_component(self, component):
        """
        Add a new component to the network experiment.
        """
        if component.name in self.components:
            raise Exception("%s is already a component of this network!"
                            % component.name)
        self.components[component.name] = component

    def dump(self, message, model_name=""):
        """
        Write components values.
        """
        for name, component in self.components.items():
            component_name = "%s_%s.mat" % (model_name, name) if model_name else "%s.mat" % name
            component_path = os.path.join(self.dump_path, component_name)
            if not hasattr(component, 'params'):
                component_values = {component.name: component.get_value()}
            else:
                component_values = {
                    param.name: param.get_value()
                    for param in component.params
                }
            scipy.io.savemat(
                component_path,
                component_values
            )
        logger.info(message)

    def load(self, model_name=""):
        """
        Load components values.
        """
        for name, component in self.components.items():
            component_name = "%s_%s.mat" % (model_name, name) if model_name else "%s.mat" % name
            component_path = os.path.join(self.dump_path, component_name)
            component_values = scipy.io.loadmat(component_path)
            if not hasattr(component, 'params'):
                param_value = component.get_value()
                assert component_values[component.name].size == param_value.size
                component.set_value(component_values[component.name].astype(np.float32))
            else:
                for param in component.params:
                    param_value = param.get_value()
                    assert component_values[param.name].size == param_value.size, (param, component_values[param.name].shape, param_value.shape)
                    param.set_value(np.reshape(
                        component_values[param.name],
                        param_value.shape
                    ).astype(np.float32))


class Sequential(object):
    """
    Create sequential networks.
    """
    def __init__(self, *modules):
        """
        Initialize a sequential network.
        """
        self.modules = [module for module in modules]

    @property
    def params(self):
        """
        Return the parameters of all objects in the sequence.
        """
        return sum([module.params for module in self.modules], [])

    def add_module(self, module):
        """
        Append a module to the sequential network.
        """
        self.modules.append(module)

    def link(self, input):
        """
        Propagate the input through the network and return the output.
        """
        for module in self.modules:
            input = module.link(input)
        return input
