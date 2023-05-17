from model import Net, get_object_detection_model
from PIL import Image
import cv2
import numpy as np
import albumentations as A
import torchvision.transforms.functional as TF
import torch.optim as optim
import torch
import argparse
from utils import *
from albumentations.pytorch.transforms import ToTensorV2
import os
from tqdm import tqdm
join = os.path.join


parser = argparse.ArgumentParser(description='Wild Rift')
parser.add_argument('--test_dir', type=str, default='/content/drive/MyDrive/Task/Egg/test_images')
parser.add_argument('--text_output_file', type=str, default='text.txt')
parser.add_argument('--check_point1', type=str, default='checkpoints/Faster.pth')
parser.add_argument('--check_point2', type=str, default='checkpoints/check.pt')
parser.add_argument('--oneshotcheck_path', type=str,default='data/oneshot_data')
args = parser.parse_args()

test_dir = args.test_dir
text_output_path = args.text_output_file
path_check1 = args.check_point1
path_check2 = args.check_point2
root_check = args.oneshotcheck_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


Detect = get_object_detection_model().to(device)
checkpoints1 = torch.load(path_check1)
Detect.load_state_dict(checkpoints1)
Detect.eval()

CheckNet = Net()
optimizer = optim.Adam(CheckNet.parameters(), lr = 0.0006)
load_checkpoint(CheckNet,optimizer,path_check2)
CheckNet.to(device)
CheckNet.eval()

transform = A.Compose([
    ToTensorV2(),
])
data = []
for i in tqdm(os.listdir(join(test_dir))):
    try:
      path = join(test_dir,i)
      mainImg = cv2.imread(path)
      mainImg = cv2.cvtColor(mainImg, cv2.COLOR_BGR2RGB).astype(np.float32)     
      mainImg = cv2.resize(mainImg, (480, 480), cv2.INTER_AREA)
      mainImg /= 255.0
      mainImg = transform(image=mainImg)['image']
      with torch.no_grad():
          prediction = Detect([mainImg.to(device)])[0]
      nms_prediction = apply_nms(prediction, iou_thresh=0.2)
      box = nms_prediction['boxes']
      box_sort = sorted(box, key=lambda x: x[0])
      image = cv2.imread(path)
      x1, y1, x2, y2 = optimus(box_sort[0],image)
      cropped_image = image[y1:y2, x1:x2]
      with torch.no_grad():
          mainImg1 = cropped_image.astype(np.uint8)
          mainImg1 = cv2.cvtColor(mainImg1, cv2.COLOR_BGR2RGB)
          mainImg1 = Image.fromarray(mainImg1)
          mainImg1 = TF.resize(mainImg1,(105,105))
          mainImg1 = TF.to_tensor(mainImg1).to(device).unsqueeze(0)
          root = join(root_check)
          result = {}
          for champ in os.listdir(root):
              avr = []
              for name in os.listdir(join(root,champ)):
                  img2 = Image.open(join(root,join(champ,name)))
                  img2 = TF.resize(img2,(105,105))
                  img2 = TF.to_tensor(img2)
                  img2 = img2.to(device).unsqueeze(0)
                  output = CheckNet(mainImg1,img2)
                  avr.append(output)
              result[champ] = Average(avr)

          max_value = max(result, key=result.get)
      data.append([i,max_value])    
    except:
      print(i)
with open(text_output_path, "w") as file:
    for line in data:
        file.write(" ".join(line) + "\n")

print("done!")
