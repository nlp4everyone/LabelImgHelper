# 🛸 Introduction:

This repository is built for developing convenient features while working with LabelImg and Ultralytics Yolo. 

<br />

# 🔑 Feature:
- Normalize dataset created from LabelImg with standard format using in Ultralytic tranining pipeline

<br />

# 🤖 Installation:
1. Clone project by typing:
```
git clone https://github.com/nlp4everyone/eve_agent
```
2. After cloning the project, install dependencies by commands:
```
pip install -r requirements.txt
```
<br />

# 🔗 Converting data from LabelImg to standard Ultralytics training data:

1. Check hinting with command 
```
python prepare_ultralytic_dataset.py -h
```
2. Place your source dataset (created from LabelImg) at the same hierarchical level with requirement file. 
3. Run this command for converting 
```
python prepare_ultralytic_dataset.py -source your_source_directory
```
<br />

# 🔗 Google Colab Auto Training
Open Google Colab Notebook and open file:
<br />
`
Auto_training_Yolo.ipynb
`
<br />
<br />

# 🔗 Auto Labeling with Ultralytics
1. Check hinting with command 
```
python auto_label.py.py -h
```
2. Run this command for auto-labeling 
```
python auto_label.py -source your_source_directory -destination your_destination_folder -train_path your_yolo_path -resize_size squared_image_size
```
