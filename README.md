# *DebFace*: Jointly De-biasing Face Recognition and Demographic Attribute Estimation

By Sixue Gong, Xiaoming Liu, and Anil K. Jain

## Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Usage](#usage)

### Introduction

This code archive includes the Python implementation of disentangling face identity and demographic attributes. Our work, *DebFace*, addressed the problem of bias in automated face recognition and demographic attribute estimation algorithms, where errors are lower on certain cohorts belonging to specific demographic groups. The proposed "DebFace" is able to extract disentangled feature representations for both face recognition and demographic estimation, so as to abate bias influence from other factors.

### Citation

If you think **DebFace** is useful to your research, please cite:

  @inproceedings{gong2020jointly,
   title={Jointly de-biasing face recognition and demographic attribute estimation},
   author={Gong, Sixue and Liu, Xiaoming and Jain, A},
   year={2020},
   organization={ECCV}
  }

**Link** to the paper: http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740324.pdf

### Usage

## More remarks
Apart from the four major classes I just mensioned before, you may need to edit the belowing files as well, to make the whole program work. All these files are in the root folder.
1. "config.py" and "args.py". These two files help to load parameters of the configuration. The former defines what kinds of parameters in the configuration, while the latter assigns the specific value to each parameter. You can assign values to any parameter in the configuration except "--result_path", where the trained models and log files will be saved. This can only be set by comman line, and you can look at the example, "train.sh" for more information.
2. "dataloader.py". You may need to add your own loader class to this file.
3. "train.py" and "test.py". You may need to change the parts of data loading, loss computation, and performance evaluation in these two files, to make them compatible with the data loader, loss method, and evaluation metric that you define.

# Storage
Once the network starts training, each currently best model will be saved so that you can stop it and then resume any time you want. The function about models and log files saving can be found in "plugins" folder.
1. **Monitor**: print loss and accuracy on the screen.
2. **Logger**:  write the loss and accuracy to log files.

# Visualization
I use a python plotting tool, Visdom, to visualize the training process. You can go to its website [Visdom](https://github.com/facebookresearch/visdom#usage) for more information.
Basically, before running this code, Visdom must be running first. It's done by:
> python -m visdom.server -p 8097
The number "8097" is an example port number for Visdom, you can set other port number as well, but be sure to claim the same prot number in the "args.py" as well.
The code for visualization is put in "plugins" folder, too.
1. **Visualizer**: visualized by visdom.
