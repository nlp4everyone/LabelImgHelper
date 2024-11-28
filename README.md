# 🛸 Introduction:

This repository is built for developing convenient features while working with LabelImg and Ultralytics Yolo. 

<br />

# 🔑 Feature:
- Normalize dataset created from LabelImg with standard format using in Ultralytic tranining pipeline

<br />

# 🔗 Converting:
1. After cloning the project, install dependencies by commands:
```
pip install -r requirements.txt
```

2. Check hinting with command 
```
python prepare_ultralytic_dataset.py -h
```
3. Place your source dataset (created from LabelImg) at the same hierarchical level with requirement file. 
4. Run this command for converting 
```
python prepare_ultralytic_dataset.py -source your_source_directory
```
