from PIL import Image
import os

# Specify the image paths
#image_path = '../../data/ILSVRC2012_v10102019/val/all/'  # Replace with your image paths
image_path = '/home/abascal/Nextcloud/spyrit_slides/spie/figures/val'

files_names = ['ILSVRC2012_val_00000003.JPEG',
               'ILSVRC2012_val_00000012.JPEG',
                'ILSVRC2012_val_00000019.JPEG',
                'ILSVRC2012_val_00000051.JPEG',
                'ILSVRC2012_val_00000056.JPEG',
               ]

image_paths = [os.path.join(image_path, file_name) for file_name in files_names]

save_folder = '/home/abascal/Nextcloud/spyrit_slides/spie/figures/target'  # Replace with your save folder

# Specify the crop points: (left, top, right, bottom)
undersample_factors = [None, 2, 2, 2, 2]
crop_points = ([220, 110], [30, 35], [30, 35], [30, 35], [30, 35])  # Replace with your crop points
img_size = (128, 128)
crop_points_list = [(point[0], point[1], point[0] + img_size[0], point[1]  + img_size[1]) for point in crop_points]
# ------------------------------------------------
def load_image_pil(image_path, undersampling_factor=None):
    # Open the image file
    img = Image.open(image_path)
    if undersampling_factor is not None:
        img = img.resize((img.size[0] // undersampling_factor, img.size[1] // undersampling_factor))
    return img

def crop_image_pil(img, crop_points):
    # Crop the image: (left, top, right, bottom)
    cropped_img = img.crop(crop_points)
    return cropped_img

# ------------------------------------------------
                
for image_path, crop_points, undersample_factor in zip(image_paths, crop_points_list, undersample_factors):
    # Open the image file
    img = load_image_pil(image_path, undersample_factor)

    # Crop the image
    cropped_img = crop_image_pil(img, crop_points)

    # Save the cropped image
    base_name = os.path.basename(image_path)
    base_name = base_name.replace('.JPEG', '_crop.JPEG')
    save_path = os.path.join(save_folder, base_name)
    # Save the cropped image
    cropped_img.save(save_path)

