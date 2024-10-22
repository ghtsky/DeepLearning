from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

img_path = "data/hymenoptera_data/train/ants/0013035.jpg"

### 使用OPENCV读取图片
# todo:使用OPENCV读取图片

# 用numpy读取图片
from PIL import Image
import numpy as np

img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

print(img_array.shape)

writer.add_image("img", img_array, 1, dataformats="HWC")

writer.close()