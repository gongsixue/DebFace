# visualizer.py

import visdom
import numpy as np
import plotly.graph_objs as go
from utils import plotlify


class Visualizer:
    def __init__(self, port, env, title):
        self.keys = []
        self.values = {}
        self.env = env
        self.viz = visdom.Visdom(port=port, env=env)
        self.iteration = 0
        self.title = title

    def register(self, modules):
        # here modules are assumed to be a dictionary
        for key in modules:
            self.keys.append(key)
            self.values[key] = {}
            self.values[key]['dtype'] = modules[key]['dtype']
            self.values[key]['vtype'] = modules[key]['vtype']
            self.values[key]['win'] = modules[key]['win'] \
                if 'win' in modules[key].keys() \
                else None
            if modules[key]['vtype'] == 'plot':
                self.values[key]['layout'] = modules[key]['layout'] \
                    if 'layout' in modules[key].keys() \
                    else {'windows': [key], 'id': 0}
                self.values[key]['value'] = []
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name=self.values[key]['layout']['windows'][i]
                ) for i in range(len(self.values[key]['layout']['windows']))]
                # Edit the layout
                layout = dict(
                    title=key,
                    xaxis=dict(title='Epoch'),
                    yaxis=dict(title=key),
                )
                fig = dict(data=data, layout=layout)
                self.values[key]['win'] = self.viz._send(
                    plotlify(fig, env=self.env, win=self.values[key]['win']))
            elif modules[key]['vtype'] == 'image':
                self.values[key]['value'] = None
            elif modules[key]['vtype'] == 'images':
                self.values[key]['value'] = None
            else:
                raise Exception('Data type not supported, please update the '
                                'visualizer plugin and rerun !!')

    def update(self, modules):
        for key in modules:
            if self.values[key]['dtype'] == 'scalar':
                self.values[key]['value'].append(modules[key])
            elif self.values[key]['dtype'] == 'image':
                self.values[key]['value'] = modules[key]
            elif self.values[key]['dtype'] == 'images':
                self.values[key]['value'] = modules[key]
            else:
                raise Exception('Data type not supported, please update the '
                                'visualizer plugin and rerun !!')

        for key in self.keys:
            if self.values[key]['vtype'] == 'plot':
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                ) for i in range(len(self.values[key]['layout']['windows']))]
                data[self.values[key]['layout']['id']] = go.Scatter(
                    x=np.array([self.iteration]).tolist(),
                    y=np.array([self.values[key]['value'][-1]]).tolist(),
                )
                fig = dict(data=data, append=True)
                self.viz._send(
                    plotlify(fig, env=self.env,
                             win=self.values[key]['win']), endpoint='update')
            elif self.values[key]['vtype'] == 'image':
                temp = self.values[key]['value'].numpy()
                for i in range(temp.shape[0]):
                    temp[i] = temp[i] - temp[i].min()
                    temp[i] = temp[i] / temp[i].max()
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.image(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
                else:
                    self.viz.image(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
            elif self.values[key]['vtype'] == 'images':
                temp = self.values[key]['value'].numpy()
                for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        temp[i][j] = temp[i][j] - temp[i][j].min()
                        temp[i][j] = temp[i][j] / temp[i][j].max()
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.images(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
                else:
                    self.viz.images(
                        temp,
                        opts=dict(
                            title=key,
                            caption=self.iteration
                        ),
                        win=self.values[key]['win']
                    )
            else:
                raise Exception('Visualization type not supported, please '
                                'update the visualizer plugin and rerun !!')
        self.iteration = self.iteration + 1
