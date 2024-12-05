import uuid

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os, torch, pathlib, argparse, time, typing

# Devine
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# model
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base",
                                             torch_dtype = torch_dtype,
                                             trust_remote_code=True).to(device)
# Processer
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base",
                                          trust_remote_code=True)
# Parser
parser = argparse.ArgumentParser(description = "Auto labeling system")
# Add argument
parser.add_argument("-source",type = str,required = True, help = "Source directory contains images")
parser.add_argument("-destination",type = str, default = "labeled_folder", help = "Destination directory you want to save the images and lables")
parser.add_argument("-resize_size",type = int,default = 640, help = "Resize image expected!")
parser.add_argument("-batch_size",type = int,default = 2, help = "Batch size")
# Parse args
args = parser.parse_args()
unlabeled_folder = args.source
labeled_folder = args.destination
resize_size = args.resize_size
batch_size = args.batch_size

# Labeled folder
if not os.path.exists(unlabeled_folder):
    raise FileNotFoundError(f"Folder path {unlabeled_folder} not existed!")

def resize_to_fix_width_size(image :Image.Image,
                             fix_width = 1024):
    aspect_ratio = image.height / image.width
    new_height = int(fix_width * aspect_ratio)

    # Step 4: Resize the image
    return image.resize((fix_width, new_height), Image.LANCZOS)

def resize_to_squared_image(image :Image.Image,
                            resize :int = 640):
    # Step 4: Resize the image
    return image.resize((resize, resize), Image.LANCZOS)

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

def predict_boxes(task_prompt :str,
                  images :typing.List[Image.Image],
                  text_input = None,
                  image_size :int = 640) -> typing.List[dict]:
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Get inout
    list_inputs = [processor(text = prompt,
                             images = element,
                             return_tensors = "pt").to(device, torch_dtype) for element in images]
    # Ids
    with torch.no_grad():
        generated_ids = [model.generate(
            input_ids = input["input_ids"],
            pixel_values = input["pixel_values"],
            max_new_tokens = 1024,
            num_beams = 3
        ) for input in list_inputs]

    # Text
    generated_values = [processor.batch_decode(id,
                                               skip_special_tokens = False)[0] for id in generated_ids]
    # Post process
    return [processor.post_process_generation(value,
                                              task = task_prompt,
                                              image_size = (image_size, image_size)) for value in generated_values]


def normalized_bbox(data :dict,resized_size :int = 640) -> typing.List[tuple]:
    boxes = data["<OCR_WITH_REGION>"]["quad_boxes"]
    results = []
    # True bboxes
    for quad_box in boxes:
        x1,y1,x2,y3 = int(quad_box[0]), int(quad_box[1]), int(quad_box[2]), int(quad_box[5])
        x_center = (x1+x2)/2
        y_center = (y1+y3)/2
        width = x2 - x1
        height = y3 - y1
        results.append((x_center/resized_size,y_center/resized_size,width/resized_size,height/resized_size))
    return results

def main():
    # Make labeled folder
    os.makedirs(labeled_folder, exist_ok = True)
    # Yolo
    image_extensions = [".png",".jpeg",".jpg"]

    # Get only image file
    image_files = [file for file in os.listdir(unlabeled_folder) if pathlib.Path(file).suffix in image_extensions]

    # Iterate
    image_paths = [os.path.join(unlabeled_folder, file_name) for file_name in image_files]
    # Get PIL Image
    pil_images = [Image.open(path) for path in image_paths]
    # Downscale
    downscaled_images = [resize_to_fix_width_size(image) for image in pil_images]
    # Squared image
    squared_images = [resize_to_squared_image(image,
                                              resize = resize_size) for image in downscaled_images]

    batch_squared_images = get_batches(squared_images,
                                       max_batch_size = batch_size)
    batch_image_paths = get_batches(image_paths,
                                    max_batch_size = batch_size)
    # Result
    batched_results = [predict_boxes(images = image,
                                     task_prompt = "<OCR_WITH_REGION>") for image in batch_squared_images]

    # Write down txt
    for (batch_result,batch_path) in zip(batched_results,batch_image_paths):
        # Get batch bbox
        batch_bboxes = [normalized_bbox(result) for result in batch_result]

        # Each element in batch
        for (bboxes,path) in zip(batch_bboxes,batch_path):
            lines_value = []
            for bbox in bboxes:
                x, y, w, h = bbox
                lines_value.append(f"0 {x} {y} {w} {h}\n")

            # Define path
            name = pathlib.Path(path).stem
            destination_class_path = os.path.join(labeled_folder, f"{name}.txt")

            # Write to file
            with open(destination_class_path, "w") as f:
                f.writelines(lines_value)

    # Write down squared image
    for (image,path) in zip(squared_images,image_paths):
        name = pathlib.Path(path).name
        destination_image_path = os.path.join(labeled_folder,name)
        image.save(destination_image_path)

    print(f"Labeling with {len(image_files)} files")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Done in total {round(end-start,2)}s")
