# Train Utility

## It is still in development stage, i.e. name and content of classes and functions in this library may be changed or removed

Train Utility is a simple utility for training in Pytorch.

## Overview

### Class

```python
ModelInfo
TrainResultInfo
TrainProcess
TrainProcess_DictLoad
TrainProcess_DictSave
TrainProcess_Folder
TrainProcess_Hook
TrainProcess_PythonFile
TrainProcess_ResultGraph
TrainProcess_ResultRecord
Interface_CodePath
INterface_DictData
```

### Function

```python
plotConfusionMatrix
plotAccuracy
plotLoss
```

## Example

### Create ModelInfo

```ModelInfo``` store all the information about the model and training. The primary usage of ```ModelInfo``` is to record all the object and data that will be used in training so we can just pass this single object instead of many parameters into training function. </br>
Below will demonstrate creating ModelInfo and filling the minimum data into the ModelInfo.

```python
# create TrainProcess
info = ModelInfo()

# fill data
info.model      = # model (nn.Module)
info.optimizer  = # optimizer (optim)

info.epoch          = 100
info.batch_size     = 2
info.learning_rate  = 1e-4
```

### Create TrainProcess and Signaling

TrainProcess will be executed when a specific stage is reached. ```TrainProcess``` is the abstract class and should not be created directly. Below is the list of some children class of TrainProcess.

- DictLoad
- DictSave
- Folder
- Hook
- PythonFile
- ResultGraph
- ResultRecord

Example will use ```TrainProcess_Folder```, which will create a folder and normally needed to be called after the training is finished and some training result need a space to store.

```python
# create TrainProcess_Folder
process = TrainProcess_Folder()

# TrainProcess_Folder will use the data in ModelInfo to create the folder
# here, it is assumed that ModelInfo (model) is created and minimum data is filled
# we need to further fill more information for TrainProcess_Folder
info.save_path      = "./Result"
info.save_folder    = "Result_xxx"

# add TrainProcess_Folder to ModelInfo
info.process_list.append(process)
```

To trigger the process, we need to call the ```ModelInfo.executeProcess(stage, info)```. There are some default stages in ```ModelInfo.Stage```.

- TRAIN_START
- ITERATION_TRAIN_START
- ITERATION_TRAIN_END
- ITERATION_VAL_START
- ITERATION_VAL_END
- TRAIN_END

Here we need to trigger ```TrainProcess_Folder``` at ```TRAIN_END``` stage.

```python
# ----- init -----
# acutally we don't need to set this
# ModelInfo.Stage.TRAIN is added to TrainProcess_Folder.stage in __init___
process.stage = [ModelInfo.Stage.END]

# ----- train end ------
# data: Dict
# what inside data is depended on different stage and user is free to add more into it
info.executeProcess(ModelInfo.Stage.END, data)
```

### Save Epoch Result

TODO: **Not yet completed**

### Plot Graph

TODO: **Not yet completed**

## Repo that Use this Library

- [https://github.com/Jack-GVDL/FastRCNN](https://github.com/Jack-GVDL/FastRCNN)
