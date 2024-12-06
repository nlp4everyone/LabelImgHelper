from ultralytics import YOLO
import os, torch, pathlib, argparse, time, typing
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
parser.add_argument("-batch_size",type = int,default = 4, help = "Resize image expected!")
# Parse args
args = parser.parse_args()
unlabeled_folder = args.source
labeled_folder = args.destination
trained_path = args.train_path
resize_size = args.resize_size
batch_size = args.batch_size

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

def resize_to_fix_size(image: torch.Tensor,
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

def get_batches(iterable :typing.List, max_batch_size :int):
    """Yield batches of max_batch_size from iterable."""
    batch = []
    for element in iterable:
        batch.append(element)
        if len(batch) >= max_batch_size:
            yield batch
            batch = []
    if batch:  # Yield any remaining elements as the last batch
        yield batch

def main():
    # Make labeled folder
    os.makedirs(labeled_folder, exist_ok = True)
    # Yolo
    yolo = YOLO(model = trained_path)
    image_extensions = [".png",".jpeg",".jpg"]

    # Get only image file
    image_files = [file for file in os.listdir(unlabeled_folder) if pathlib.Path(file).suffix in image_extensions]
    # Get path
    image_paths = [os.path.join(unlabeled_folder, file_name) for file_name in image_files]
    batch_images = get_batches(image_paths,
                               max_batch_size = batch_size)

    # Iterate each batch
    for batch_image in batch_images:
        # images
        tensor_images = [read_image(file_path) for file_path in batch_image]
        # Downscale
        downscaled_images = [resize_to_fix_size(image) for image in tensor_images]
        # Convert to squared image
        squared_images = [resize_to_square_image(image,
                                                 size = resize_size) for image in downscaled_images]
        # Stack images
        stacked_images = torch.stack(squared_images, dim = 0)
        # Normalized image
        normalized_image = stacked_images.float()/255.0

        # Predict
        results = yolo.predict(normalized_image)
        # For each result
        for (index,result) in enumerate(results):
            # Bbox
            bboxes = result.boxes.xywh.int().tolist()
            # Predicted
            predicted_class = result.boxes.cls.int().tolist()

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

            for image_path in batch_image:
                image_full_name = pathlib.Path(image_path).name
                image_name = pathlib.Path(image_path).stem
                # Define full path
                dest_image_path = os.path.join(labeled_folder,image_full_name)
                # Save image
                save_image(tensor = squared_images[index].float()/255.0,
                           fp = dest_image_path)

                # Write txt
                des_class_path = os.path.join(labeled_folder,f"{image_name}.txt")
                # Write
                with open(des_class_path,"w") as f:
                    f.writelines(write_lines)

    print(f"Labeling with {len(image_files)} files")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Done in total {round(end-start,2)}s")