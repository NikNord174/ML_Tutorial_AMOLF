# Tutorial for ML community at AMOLF
This repository is aimed to show the simpliest training process of Neural Networks using PyTorch.

## Quick start

In VSCode terminal download the repository to your computer with the command
```
git clone https://github.com/NikNord174/ML_Tutorial_AMOLF.git
```

Then, go to the ML_Tutorial_AMOLF folder and depends on your OS, follow the steps:

### For Windows users:

1. Create and activate an environment
```
python -m venv venv
venv/Scripts/activate
```
(venv) should appear at the beginning of the row in the terminal.
2. Install requirements
```
pip install -r requirements.txt
```
3. To start training
```
python main.py
```
or go to step-by-step.ipynb for more detailed description of training process.
To play in step-by-step.ipynb 
1. open the file in the same VS Code window and start the first cell with imports.
2. At the first time VS Code will suggest you to choose Python environment.
Choose (venv).
3. Then agree to install the jupyter kernel.
4. Start the cell again if necessary.

### For Mac users:

1. Create and activate an environment
```
python3 -m venv venv
source venv/bin/activate
```
2. Install requirements
```
pip3 install -r requirements.txt
```
3. To start training
```
python3 main.py
```
or go to step-by-step.ipynb for more detailed description of training process.

## Experiment with different models or hyperparameters
To try training process with different model change model argument in main.py.
Other hyperparameters are available in constants.py.
