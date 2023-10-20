# lcm-webui
Latent Consistency Models webui

A very simple webui for [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model).

![image](https://github.com/yownas/lcm-webui/assets/13150150/f8d4c745-323a-4dae-a24e-9d667cd89db2)

Example on a GTX1080ti:

![image](https://github.com/yownas/lcm-webui/assets/13150150/34e3fa35-c0f0-4625-93e1-5b4c842f0746)

# Installation

Create a venv and install requirements.

`python -m venv venv`

`source venv/bin/activate` (Linux) or `source venv\Scripts\activate.bat` (Windows)

`pip install -r requirements.txt`

# Launch

Activate venv if you haven't done it.

`source venv/bin/activate` (Linux) or `source venv\Scripts\activate.bat` (Windows)

`python webui.py`

Models will be downloaded automatically into your local hugginface cache folder.
