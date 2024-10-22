import PIL.Image
from torchvision import transforms

# toTenor

img_path = "../data/hymenoptera_data/train/ants/0013035.jpg"

img_PIL = PIL.Image.open(img_path)

tensor_trans = transforms.ToTensor()

tensor_img = tensor_trans(img_PIL)

print(tensor_img)




