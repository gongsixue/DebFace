# monitor.py

from collections import OrderedDict


class Monitor:
    def __init__(self, smoothing=False, smoothness=0.7):
        self.keys = []
        self.values = {}
        self.smoothing = smoothing
        self.smoothness = smoothness
        self.num = 0

    def register(self, modules):
        # here modules are assumed to be a dictionary
        for key in modules:
            self.keys.append(key)
            self.values[key] = {}
            self.values[key]['dtype'] = modules[key]['dtype']

            if modules[key]['dtype'] == 'running_mean':
                self.values[key]['value'] = 0
            else:
                raise Exception('Data type not supported, please update the '
                                'monitor plugin and rerun !!')

    def reset(self):
        self.num = 0
        for key in self.keys:
            self.values[key]['value'] = 0

    def update(self, modules, batch_size):
        for key in modules:
            if self.values[key]['dtype'] == 'running_mean':
                if not self.smoothing:
                    self.values[key]['value'] = (self.values[key][
                                                 'value'] * self.num +
                                                 modules[key] * batch_size
                                                 ) / (self.num + batch_size)
                else:
                    self.values[key]['value'] = self.values[key][
                        'value'] * self.smoothness + \
                        modules[key] * (1 - self.smoothness)
            else:
                raise Exception('Data type not supported, please update the '
                                'monitor plugin and rerun !!')

        self.num += batch_size

    def getvalues(self, key=None):
        if key is not None:
            return self.values[key]['value']
        else:
            return OrderedDict(
                [(key, self.values[key]['value']) for key in self.keys])
