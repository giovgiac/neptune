# Neptune

A unified codebase, programmed in Python 3, for executing a collection of neural networks being developed 
at [FURG](https://www.furg.br) by two Computer Engineering students:
* Mr. Giovanni Gatti De Giacomo
* MSc Matheus Machado dos Santos

## Networks

In the codebase there are three supported applications that run on neural networks:
* Semantic segmentation of water and stationary and moving objects in satellite images (U-Net);
* Matching between acoustic and satellite images (DizygoticNet);
* General translation of a large satellite image and acoustic image pair to a single equivalent 
satellite image (W-Net).

## Dependencies

Currently, our codebase depends on the following packages:
* Abseil >= 0.7.1
* Namegenerator >= 1.0.6
* Pandas >= 0.24.2
* TensorFlow >= 2.0.0

## Execution

To choose the neural network and dataset combination to use, the file 'main.py' needs to be edited. If you need to
change default configuration, such as shape or batch size, then you will also want to edit 'config.py'.

After setting up the appropriate networks, datasets and configurations, execution is as simple as:
```shell script
$ python main.py
```

The command for tensorboard will be automatically generated and printed by the program in the terminal.

Configuration can also be specified via command-line arguments. For a list of all possibilities take a look at
'config.py', an example is provided below:
```shell script
$ python main.py --batch_size=16 --learning_rate=1e-4
```
