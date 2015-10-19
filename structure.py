import os
import numpy as np
import scipy.io
from datetime import datetime


class Structure(object):
    """
    Network structure. For easy saving and loading.
    """
    def __init__(self, name, dump_path='/home/guillaume/workspace/saved_models'):
        self.name = name
        self.start_time = datetime.now()
        self.dump_path = os.path.join(dump_path, name)
        self.components = {}
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)

    def add_component(self, component):
        if component.name in self.components:
            raise Exception("%s is already a component of this network!"
                            % component.name)
        self.components[component.name] = component

    def dump(self, message):
        # Write components values
        for name, component in self.components.items():
            component_path = os.path.join(self.dump_path, "%s.mat" % (name))
            scipy.io.savemat(
                component_path,
                {param.name: param.get_value() for param in component.params}
            )
        # Write message
        messages_path = os.path.join(self.dump_path, "messages.txt")
        with open(messages_path, "a") as f:
            f.write("%s (%s) - %s\n" % (
                str(datetime.now()),
                str(datetime.now() - self.start_time),
                message
            ))
        # Write parameters shapes
        params_shapes_path = os.path.join(self.dump_path, "params_shapes.txt")
        with open(params_shapes_path, "w") as f:
            for name, component in self.components.items():
                for param in component.params:
                    f.write("%s__%s : %s\n"
                            % (name, param.name, param.get_value().shape))

    def load(self):
        # Load components values
        for name, component in self.components.items():
            component_path = os.path.join(self.dump_path, "%s.mat" % (name))
            component_values = scipy.io.loadmat(component_path)
            for param in component.params:
                param.set_value(np.reshape(
                    component_values[param.name],
                    param.get_value().shape
                ))


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
