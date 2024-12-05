import os, torch, pathlib, argparse, time
import uuid

from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype,
                                                 trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

# Parser
parser = argparse.ArgumentParser(description = "Auto labeling system")
# Add argument
parser.add_argument("-source",type = str,required = True, help = "Source directory contains images")
parser.add_argument("-destination",type = str, default = "labeled_folder", help = "Destination directory you want to save the images and lables")
# Parse args
args = parser.parse_args()
unlabeled_folder = args.source
labeled_folder = args.destination

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

def predict_boxes(task_prompt :str,
                  image :Image.Image,
                  text_input = None,):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

def normalized_bbox(data :dict,resized_size :int = 640):
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
    for file_name in image_files:
        # Define file path
        file_path = os.path.join(unlabeled_folder, file_name)
        image = Image.open(file_path)
        # Downscale
        resized_image = resize_to_fix_width_size(image)
        # Resize to squared image
        squared_image = resize_to_squared_image(resized_image,
                                                resize = 640)
        # Predict boxes
        result = predict_boxes(image = squared_image, task_prompt ="<OCR_WITH_REGION>")
        xywh = normalized_bbox(result)

        write_lines = []
        for i in range(len(xywh)):
            x,y,w,h = xywh[i]
            # add class
            line = f"0 {x} {y} {w} {h}\n"
            write_lines.append(line)
        # Save images
        des_file_path = os.path.join(labeled_folder,file_name)

        name = pathlib.Path(file_name).stem
        # Write txt
        destination_class_path = os.path.join(labeled_folder,f"{name}.txt")
        # Write txt
        with open(destination_class_path, "w") as f:
            f.writelines(write_lines)
            
        # Save image
        squared_image.save(des_file_path)

    print(f"Labeling with {len(image_files)} files")

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Done in total {round(end-start,2)}s")
