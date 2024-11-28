from pathlib import Path
from typing import List, Literal
import os, random, shutil, time, yaml, argparse

# Parser
parser = argparse.ArgumentParser(description = "Converting yaml")
# Add argument
parser.add_argument("-source",type = str,required = True, help = "Source directory")
parser.add_argument("-destination",type = str, default = "output", help = "Source directory")
parser.add_argument("-train_percent",type = float,default = 0.9, help = "Train percent")
parser.add_argument("-test_percent",type = float,default = 0.05, help = "Test percent")
parser.add_argument("-val_percent",type = float,default = 0.05, help = "Val percent")
args = parser.parse_args()
source_dir = args.source
destination_dir = args.destination
train_percent = args.train_percent
test_percent = args.test_percent
val_percent = args.val_percent

# Define extensions
image_extensions = [".jpg",".png",".jpeg"]
label_extenstions = [".txt"]
# Random seed
random.seed(random.randint(1,100))

class YamlConfiguration:
    @staticmethod
    def load_config(config_path :str = "data.yaml") -> dict:
        if not os.path.exists(config_path):
            raise Exception(f"File: {config_path} not found")

        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def save_config(data :dict,
                    config_file :str = "data.yaml",
                    class_file :str = "classes.txt",
                    source_dir :str = args.source,
                    des_dir :str = args.destination):
        # Class path
        classes_path = os.path.join(source_dir,class_file)
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Class path: {classes_path} not existed!")
        # Load classes
        with open(classes_path,"r") as f:
            classes_list = f.readlines()
            # Remove \n from list
            classes_list = [element.replace("\n","").strip() for element in classes_list]

        # Update value
        data.update({"names": classes_list})
        data.update({"nc": len(classes_list)})

        # Save config
        with open(os.path.join(des_dir,config_file), 'w') as file:
            yaml.dump(data, file)

def move_file(images_data :List[str],
              labels_data :List[str],
              mode :Literal["train","test","val"] = "train",
              des_dir :str = args.destination):
    if len(images_data) != len(labels_data):
        raise Exception("Wrong")
    # Make dir
    os.makedirs(os.path.join(des_dir,mode))

    for (image_path, label_path) in zip(images_data,labels_data):
        # Get base name
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)
        # Get des path
        des_image_path = os.path.join(des_dir,mode,image_name)
        des_label_path = os.path.join(des_dir,mode,label_name)
        # Copy
        shutil.copy(image_path,des_image_path)
        shutil.copy(label_path,des_label_path)

def contruct_dataset(shuffled_label_files :List[str],
                     shuffled_image_files :List[str]):
    # Get element each
    num_train = int(len(shuffled_label_files) * train_percent)
    num_test = int(len(shuffled_label_files) * test_percent)

    # Image
    train_images = shuffled_image_files[:num_train]
    test_images = shuffled_image_files[num_train:num_train + num_test]
    val_images = shuffled_image_files[num_train + num_test:]

    # Label
    train_labels = shuffled_label_files[:num_train]
    test_labels = shuffled_label_files[num_train:num_train + num_test]
    val_labels = shuffled_label_files[num_train + num_test:]

    # Move train images and labels
    move_file(train_images, train_labels)
    # Move test images and labels
    move_file(test_images,test_labels,mode = "test")
    # Move val images and labels
    move_file(val_images, val_labels, mode = "val")

    # Load and save config
    data_config = YamlConfiguration.load_config()
    # Insert data
    data_config.update({"train": os.path.join(destination_dir,"train")})
    data_config.update({"test": os.path.join(destination_dir, "test")})
    data_config.update({"val": os.path.join(destination_dir, "val")})
    # Save config
    YamlConfiguration.save_config(data = data_config)

def compress_folder(source_folder_path :str):
    compressed_folder_path = source_folder_path + ".zip"
    shutil.make_archive(source_folder_path,'zip',source_folder_path)


def main():
    # Check source path
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source folder: {source_dir} not found!")

    # Percent
    if train_percent + test_percent + val_percent != 1:
        raise ValueError(f"Total percent must be equal to 1. Train percent: {train_percent}, test percent: {test_percent}, val percent: {val_percent}")

    # Delete current output directory
    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    # Create output directory if not existed
    os.makedirs(destination_dir)

    # List of images/labels
    image_files = [os.path.join(source_dir,file_path) for file_path in os.listdir(source_dir) if Path(file_path).suffix in image_extensions]
    label_files = [os.path.join(source_dir,file_path) for file_path in os.listdir(source_dir) if Path(file_path).suffix in label_extenstions and not file_path.startswith("classes")]

    # Check amounts images and labels
    if len(image_files) != len(label_files):
        raise ValueError(f"Difference between number of images ({len(image_files)}) and labels ({len(label_files)})")

    # Combine both lists using zip
    combined = list(zip(sorted(image_files), sorted(label_files)))
    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip back into two separate lists
    shuffled_image_files, shuffled_label_files = zip(*combined)
    # Contruct dataset
    contruct_dataset(shuffled_label_files = shuffled_label_files,
                     shuffled_image_files = shuffled_image_files)
    # Print status
    print(f"Copy done with total :{len(image_files)} pairs")
    # Compress
    compress_folder(source_folder_path = destination_dir)
    print(f"Output saved as {destination_dir}.zip")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    # Print
    print(f"Processing in {round(end- start,5)}s")