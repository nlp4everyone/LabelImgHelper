from ultralytics import YOLO
import os, torch, pathlib, argparse, time
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
# Parser
parser = argparse.ArgumentParser(description = "Auto labeling system")
# Add argument
parser.add_argument("-source",type = str,required = True, help = "Source directory contains images")
parser.add_argument("-train_path",type = str, required = True, help = "Path to your train model")
parser.add_argument("-destination",type = str, default = "labeled_folder", help = "Destination directory you want to save the images and lables")
parser.add_argument("-resize_size",type = int,default = 640, help = "Resize image expected!")
# Parse args
args = parser.parse_args()
unlabeled_folder = args.source
labeled_folder = args.destination
trained_path = args.train_path
resize_size = args.resize_size

# Check path
if not os.path.exists(trained_path):
    raise FileNotFoundError(f"Trained path {trained_path} not existed!")
# Check pytorch model
model_extension = pathlib.Path(trained_path).suffix
if model_extension != ".pt":
    raise ValueError("Trained model must be end with Pytorch extension (.pt)")

# Labeled folder
if not os.path.exists(unlabeled_folder):
    raise FileNotFoundError(f"Folder path {unlabeled_folder} not existed!")

def resize_to_square_image(image: torch.Tensor,
                           size: int) -> torch.Tensor:
    """
    Resize image to squared format
    :param image:
    :param size:
    :return:
    """
    # Get total dimension
    dim = image.ndim

    accept_shapes = [2, 3, 4]
    # Case BGR dimension
    if dim not in accept_shapes:
        raise ValueError("Input image has wrong dimension")
    # Define resize function
    resize_transform = transforms.Resize(size=(size, size))
    # Resize to width resize if above
    return resize_transform(image)

def resize_to_fix_size_v2(image: torch.Tensor,
                          downscale_width: int = 1024) -> torch.Tensor:
    """
    Resize to fix size ver 2
    :param image:
    :param downscale_width:
    :return:
    """
    # Get total dimension
    dim = image.ndim
    # Case BGR dimension
    if dim == 3:
        # Get height, width and chanel
        c, h, w = image.shape
    elif dim == 2:
        # Gray case
        h, w = image.shape
    else:
        raise ValueError("Input image has wrong dimension")
    # When width below width resize, return itself
    if w < downscale_width:
        return image
    # Calculate the new height to maintain the aspect ratio
    aspect_ratio = h / w
    new_height = int(downscale_width * aspect_ratio)
    # Define resize function
    resize_transform = transforms.Resize(size=(new_height, downscale_width))
    # Resize to width resize if above
    return resize_transform(image)

def main():
    # Make labeled folder
    os.makedirs(labeled_folder, exist_ok = True)
    # Yolo
    yolo = YOLO(model = trained_path)
    image_extensions = [".png",".jpeg",".jpg"]

    # Get only image file
    image_files = [file for file in os.listdir(unlabeled_folder) if pathlib.Path(file).suffix in image_extensions]
    # Iterate
    for file_name in image_files:
        # Define file path
        file_path = os.path.join(unlabeled_folder, file_name)
        # image tensor
        image_tensor = read_image(file_path)
        # Resize to fix size
        image_tensor = resize_to_fix_size_v2(image_tensor)
        # Resize to square image
        squared_image_tensor = resize_to_square_image(image_tensor, size = resize_size)
        squared_image_tensor = squared_image_tensor.float() / 255.0

        # Predict
        results = yolo.predict(squared_image_tensor)
        # Bbox
        bboxes = results[0].boxes.xywh.int().tolist()
        class_dict = results[0].names
        predicted_class = results[0].boxes.cls.int().tolist()

        write_lines = []
        for i in range(len(predicted_class)):
            # Define coordination
            x,y,w,h = tuple(bboxes[i])
            # Scaled
            x_scaled, y_scaled, w_scaled, h_scaled = x/resize_size, y/resize_size, w/resize_size, h/resize_size
            # Define class
            cls = predicted_class[i]
            line = f"{cls} {x_scaled} {y_scaled} {w_scaled} {h_scaled}\n"
            write_lines.append(line)

        # Save images
        des_file_path = os.path.join(labeled_folder,file_name)
        save_image(tensor = squared_image_tensor,
                   fp = des_file_path)

        name = pathlib.Path(file_name).stem
        # Write txt
        destination_class_path = os.path.join(labeled_folder,f"{name}.txt")
        # Write
        with open(destination_class_path,"w") as f:
            f.writelines(write_lines)
    print(f"Labeling with {len(image_files)} files")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Done in total {round(end-start,2)}s")