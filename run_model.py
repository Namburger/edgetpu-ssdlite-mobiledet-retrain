import os
import sys
import time
import numpy as np
import platform
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image
from PIL import ImageDraw

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

if len(sys.argv) < 3:
  print('Usage:', sys.argv[0], '<model_path> <test_image_dir>')
  exit()

model_path = str(sys.argv[1])

# Creates tflite interpreter
if 'edgetpu' in model_path:
  interpreter = Interpreter(model_path, experimental_delegates=[
      load_delegate(EDGETPU_SHARED_LIB)])
else:
  interpreter = Interpreter(model_path)

interpreter.allocate_tensors()
interpreter.invoke()  # warmup
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]


def run_inference(interpreter, image):
  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()
  boxes = interpreter.get_tensor(output_details[0]['index'])[0]
  classes = interpreter.get_tensor(output_details[1]['index'])[0]
  scores = interpreter.get_tensor(output_details[2]['index'])[0]
  # num_detections = interpreter.get_tensor(output_details[3]['index'])[0]
  return boxes, classes, scores


t = 0
test_image_paths = [os.path.join(str(sys.argv[2]) + '/image{}.jpg'.format(i)) for i in range(1, 9)]
for image_path in test_image_paths:
  print('Evaluating:', image_path)
  image = Image.open(image_path)
  image_width, image_height = image.size
  draw = ImageDraw.Draw(image)
  resized_image = image.resize((width, height))
  np_image = np.asarray(resized_image)
  input_tensor = np.expand_dims(np_image, axis=0)
  # Run inference
  t0 = time.perf_counter()
  boxes, classes, scores = run_inference(interpreter, input_tensor)
  t += time.perf_counter() - t0
  # Draw results on image
  colors = {0: (128, 255, 102), 1: (102, 255, 255)}
  labels = {0: 'abyssian cat', 1: 'american bulldog'}
  for i in range(len(boxes)):
    if scores[i] > .7:
      ymin = int(max(1, (boxes[i][0] * image_height)))
      xmin = int(max(1, (boxes[i][1] * image_width)))
      ymax = int(min(image_height, (boxes[i][2] * image_height)))
      xmax = int(min(image_width, (boxes[i][3] * image_width)))
      draw.rectangle((xmin, ymin, xmax, ymax), width=7,
                     outline=colors[int(classes[i])])
      draw.rectangle((xmin, ymin, xmax, ymin-10),
                     fill=colors[int(classes[i])])
      text = labels[int(classes[i])] + ' ' + str(scores[i]*100) + '%'
      draw.text((xmin+2, ymin-10), text, fill=(0, 0, 0), width=2)
  image.show()
print('Inference time: ', t)
