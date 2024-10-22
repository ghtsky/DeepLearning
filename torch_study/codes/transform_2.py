from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from  torchvision import transforms

writer = SummaryWriter("../logs")

img = Image.open("../data/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg")

print(img)

STEP = 1

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor, global_step=1)

#========= Normalization 归一化
# input[channel] = (input[channel] - mena[channel]) / std[channel]

trans_norm = transforms.Normalize([0.1,0.1,0.1], [0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)

writer.add_image("Norm", img_norm, global_step=1)
# print(img_tensor - img_norm)
#
# print()



#========= Resize

trans_resize = transforms.Resize((512,512))

img_resize = trans_resize(img_tensor)

writer.add_image("Resize", img_resize, global_step=1)


#========= Compose -resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])

img_resize_2 = trans_compose(img)

writer.add_image("Resize", img_resize_2, global_step=2)


#========= Random Crop










writer.close()