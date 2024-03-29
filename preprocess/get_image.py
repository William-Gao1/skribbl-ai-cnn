import numpy as np
from PIL import Image, ImageDraw
import math

def strokes_to_bitmap(strokes, width, height, stroke_width):
    # Create a blank white image
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)

    #Draw strokes on the image
    for stroke in strokes:
        draw.line(list(zip(stroke[0], stroke[1])), fill=255, width=stroke_width)
    
    return image

def get_bbox(strokes):
  max_x = 0
  min_x = strokes[0][0][0]
  max_y = 0
  min_y = strokes[0][1][0]
  
  for stroke in strokes:
    max_x = max(np.max(stroke[0]), max_x)
    max_y = max(np.max(stroke[1]), max_y)
    min_x = min(np.min(stroke[0]), min_x)
    min_y = min(np.min(stroke[1]), min_y)
  
  return math.ceil(min_y), math.ceil(max_y), math.ceil(min_x), math.ceil(max_x)

def get_img_from_strokes(strokes, target_shape=(64,64)):
  min_y, max_y, min_x, max_x = get_bbox(strokes)
  
  stroke_width = math.ceil(max(max_y - min_y, max_x - min_x) / (3.5 * math.sqrt(target_shape[0])))
  
  half_stroke_width = math.ceil(stroke_width/2)

  image_data = np.array(strokes, dtype=object)

  # add some margin to account for width of strokes when using PIL to draw the image
  bitmap_image = strokes_to_bitmap(image_data, max_x + half_stroke_width, max_y + half_stroke_width, stroke_width)  

  # crop to bounding box and then resize
  cropped = bitmap_image.crop((min_x - half_stroke_width, min_y - half_stroke_width, max_x + half_stroke_width, max_y + half_stroke_width))
  cropped.thumbnail(target_shape, resample=Image.BICUBIC)
  thumb = np.array(cropped)
  
  padded = np.pad(thumb, ((math.floor((target_shape[0] - thumb.shape[0])/2), math.ceil((target_shape[0] - thumb.shape[0])/2)), (math.floor((target_shape[0] - thumb.shape[1])/2), math.ceil((target_shape[0] - thumb.shape[1])/2))))
  
  normalized = (padded - np.mean(padded))/np.std(padded)
  
  return normalized.astype(np.float32)