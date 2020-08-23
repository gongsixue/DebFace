# How to use it?
It is easy to be customized to fulfill your own demands. All you need to do is to create the necessary classes which I list below:
 1. **Data Loader**: defines how you load your training and testing data.
 For your information, I put some data loaders I often use in the "datasets" folder, such as "folderlist.py" (loading images in a given folder), "filelist.py" (loading images in a file list written in a text file), "triplet.py" (loading images as triplets), and so on.
 2. **Network Architecture**: uses a nn.Module class to define your neural network.
 I put some well-kown networks in face recognition domain in the "models" folder, such as "resnet.py" (ResNet), "preactresnet.py" (PreAct-ResNet), and "sphereface.py" (SphereFace).
 3. **Loss Method**: Pytorch provides some comman loss functions in torch.nn library, for example CrossEntropyLoss or MSELoss. And you can also design your own loss function by using a nn.Module class, which is similary to writting a network architecture.
 Still, there are some loss functions I wrote for face recognition and shoe image retrieval, that can be found in "losses" folder.
 4. **Evaluation Metric**: a class to measure the accuracy (performance) of your network model, i.e., how to test your network.
 Again, I show examples in the "evaluate" folder.

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
