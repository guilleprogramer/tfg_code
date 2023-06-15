import os
import numpy as np
import cv2
import requests
from fastapi import FastAPI, Request
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from starlette.responses import StreamingResponse
from torchvision.transforms.functional import normalize

from models import ISNetDIS


def run_CNN(image_path):
    output_folder = "test_folder"
    model_path = "IS-Net/3jun.pth"  # model route
    input_size = [1024, 1024]
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    im = io.imread(image_path)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_shp = im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    if torch.cuda.is_available():
        image = image.cuda()
    result = net(image)
    result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)

    im_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, im_name + ".png")

    # Get the mask
    mask = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)

    # Multiply the final image by the original image
    blended_image = (im.astype(float) * result.permute(1, 2, 0).cpu().data.numpy()).astype(np.uint8)

    # Add alpha channel to merged image
    blended_image_with_alpha = np.concatenate((blended_image, mask), axis=2)

    return blended_image_with_alpha, output_path


def superimpose_images(image_bg, image_fg, position):
    # Gets the dimensions of both images
    height_bg, width_bg, _ = image_bg.shape
    height_fg, width_fg, _ = image_fg.shape

    # Make sure that the image to be overlaid does not exceed the limits of the background image
    height = min(height_bg, height_fg)
    width = min(width_bg, width_fg)

    # Calculate the overlap limits on the background image
    start_y = position[1]
    end_y = start_y + height
    start_x = position[0]
    end_x = start_x + width

    # Create a new resulting image with the appropriate dimensions
    result = np.copy(image_bg)

    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            try:
                # Gets the pixel values in each image
                pixel_bg = image_bg[y, x]
                pixel_fg = image_fg[y - start_y, x - start_x]

                # Gets the value of the alpha channel
                alpha = pixel_fg[3] / 255.0

                # Calculate the value of the overlapping color
                color = (1 - alpha) * pixel_bg[:3] + alpha * pixel_fg[:3]

                # Assigns the new color value to the resulting image
                result[y, x, :3] = color.astype(np.uint8)
            except:
                pass

    return result


def resize_image(image, scale_percent):
    # Gets the dimensions of the original image
    height, width = image.shape[:2]

    # Calculate new image size based on percentage
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)

    # Resize the image using the new size
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


def generate_bounding_box(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the contours in the image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest outline (corresponding to the car)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the coordinates of the corners of the bounding box
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_left = (x, y + h)
    bottom_right = (x + w, y + h)

    return top_left, top_right, bottom_left, bottom_right


def overlay_car(image_bg, image_fg):
    top_left, top_right, bottom_left, bottom_right = generate_bounding_box(image_fg)

    # Calculate the distance between the lower left and right points of the bounding box
    distance = np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))

    # Calculate the scale factor
    scale_percent = (1100 / distance) * 100

    image_fg = resize_image(image_fg, scale_percent)
    top_left, top_right, bottom_left, bottom_right = generate_bounding_box(image_fg)

    # Calculate the coordinates of the upper left and lower right points of the bounding box
    top_left = (min(top_left[0], bottom_left[0]), min(top_left[1], top_right[1]))
    bottom_right = (max(bottom_right[0], bottom_left[0]), max(bottom_right[1], top_right[1]))
    top_left = [top_left[0] + 700, top_left[1] + 1000]
    bottom_right = [bottom_right[0] + 700, bottom_right[1] + 1000]

    # Calculate the width and the height of the bounding box
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    shadow = cv2.imread("sombra2.png", cv2.IMREAD_UNCHANGED)
    shadow = cv2.resize(shadow, (width + 1200, height + 500))

    image_bg = superimpose_images(image_bg, shadow, (500, 1000))
    result = superimpose_images(image_bg, image_fg, (500, 900))

    return result


app = FastAPI()


@app.get("/items/{image_path}")
def api(request: Request, image_path: str):
    image_path = request.path_params["image_path"]
    image_bg = cv2.imread("fondo_flexicar.png")
    image_fg, output_path = run_CNN(image_path)

    result = overlay_car(image_bg, image_fg)

    # Save the resulting image with transparency
    cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    res, im_png = cv2.imencode(".png", result)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


if __name__ == "__main__":

    resp = requests.get('http://127.0.0.1:8000/items/dani_test/IMG_5447.JPG')
