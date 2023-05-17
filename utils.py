import torch
import torchvision
import os
join = os.path.join


def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction
def load_checkpoint(model, optimizer,path):
    save_path = join(path)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(f'Model loaded from <== {save_path}')
    
    return val_loss
def ab(x):
  return torch.tensor(x, device = 'cpu')

def optimus(a,b):
  wt = b.shape[1]
  ht = b.shape[0]

  x_min = int(a[0]*wt/480)
  x_max = int(a[2]*wt/480)
  y_min = int(a[1]*ht/480)
  y_max = int(a[3]*ht/480)

  return x_min, y_min, x_max, y_max

def Average(lst):
    return sum(lst) / len(lst)