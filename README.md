# Project Control Estimation Module

Project Control effort and estimated time estimation module for the PROMESSA project. Uses machine learning models to estimate the effort and delivery time of PROJECT CONTROL tasks.

### Requirements

* [Python 3.6+](https://www.python.org/)
* [pip](https://pypi.org/project/pip/)

### Usage

First, install the project's dependencies using pip:

```
$ pip install -r requirements.txt
```

Afterwards, start up the server through the application's entry script:

```
$ python main.py
```

What the entry script does is load the required modules and bring the Flask server online, which makes the REST API (and respective documentation) available for use.

The REST API documentation can be accessed through http://localhost:8080/docs.

### Package/Module setup fix

Due to the project being set up as a reusable package with its respective modules (e.g. model, utils in the modules folder) for integration with the main PROMESSA module, this can lead to a **ModuleNotFoundError** (No module named 'effort_estimation') when running the ```$ python main.py``` command on a Windows environment. To fix this, run the following command in the terminal before running ```$ python main.py```:

```
$ set PYTHONPATH=%PYTHONPATH%;<path\to\the\project\folder>
```

Example:

```
set PYTHONPATH=%PYTHONPATH%;F:\Daniel\INESTEC-AE2021-0085\Daniel\project-control-effort-estimation
```

### Folder Structure Explanation

**data** - folder used to store the spreadsheets used to train the model and for analysis of the data used.

**modules** - this folder isolates the different functionalities of the application to separate modules. It contains a file for the model logic (`model.py`), one of the REST API and Flask server logic (`server.py`), and another to store utility methods and others that are commonly reused throughout the application (`utils.py`).

**scripts** - this is similar to the modules folder as it also stores various Python scripts, but these are isolated scripts for things like testing new code logic before it is added to the application itself. The difference is that the files in the modules folder are directly used in the application, whereas these scripts do not have any effect on the application when it is running.

**storage** - folder used for persistent storage of the trained models, encoders, and anything else that needs to be re-used after the models have been trained.

**web** - used to store static content served over the Flask server, such as the REST API and its respective documentation.