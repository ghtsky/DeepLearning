import PIL.Image
import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# toTenor

img_path = "../data/hymenoptera_data/train/ants/0013035.jpg"

tensor_trans = transforms.ToTensor()

#PIL类型
img_PIL = PIL.Image.open(img_path)

tensor_img = tensor_trans(img_PIL)

#ndarray类型
img_cv = cv2.imread(img_path)

tensor_img2 = tensor_trans(img_cv)

writer = SummaryWriter(log_dir='./logs')

writer.add_image("TensorImg", tensor_img, global_step=1)

writer.close()

