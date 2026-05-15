# Web Style Transfer

## Introduction

The purpose of this app is to perform style transfer between images within the browser environment, using WebGPU. As an entrypoint, we have a working python implementation that is as trimmed down as possible, and should be used as reference for all the required operations needed in our WebGPU implementation.

## Running Python Version

From the root folder, first install dependencies:

```
pip install -r requirements.txt
```

Then, run the script with:

```
python python-reference/style-transfer.py
```

This will generate a folder `./expt` with the outputs from style transfer.
