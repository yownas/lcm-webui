# lcm-webui
Latent Consistency Models webui

A very simple webui for [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model).

![image](https://github.com/yownas/lcm-webui/assets/13150150/f8d4c745-323a-4dae-a24e-9d667cd89db2)

Example on a GTX1080ti:

![image](https://github.com/yownas/lcm-webui/assets/13150150/34e3fa35-c0f0-4625-93e1-5b4c842f0746)

# Installation

Create a venv and install requirements.

Install python from here (https://www.python.org/downloads/) (Tested with version 3.10). Download this repo with git (recommended) or as a zip file (links in the green "Code" button at the top of the page). Open a cmd (Windows) or bash (Linux) prompt and go to the folder containing the webui and create a virtual env for python.

`python -m venv venv`

Activate it.

`source venv/bin/activate` (Linux) or `source venv\Scripts\activate.bat` (Windows)

Install pytorch from (https://pytorch.org). Select Stable, your OS, Pip, Python and compute platform that match your computer.

Install the rest of the requirements.

`pip install -r requirements.txt`

# Launch

Activate venv if you haven't done it.

`source venv/bin/activate` (Linux) or `source venv\Scripts\activate.bat` (Windows)

`python webui.py`

Go to (http://localhost:7860)

Models will be downloaded automatically into your local hugginface cache folder.
