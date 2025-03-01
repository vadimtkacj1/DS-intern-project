import os
from PIL import Image

def filter_images(data_path, image_exts = ['jpeg', 'jpg', 'png']):
    for image_class in os.listdir(data_path): 
        for image in os.listdir(os.path.join(data_path, image_class)):
            image_path = os.path.join(data_path, image_class, image)
            try: 
                with Image.open(image_path) as img:
                    tip = img.format.lower()
                    if tip not in image_exts:
                        print('Image not in ext list {}'.format(image_path))
                        os.remove(image_path)
            except OSError:
                print(f'Could not read file: {image_path}')
                os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))

def translate_dir_names(data_path, translate):
    for image_class in os.listdir(data_path): 
        if image_class in translate:
            current_dir = os.path.join(data_path, image_class)
            new_dir = os.path.join(data_path, translate[image_class])
            os.rename(current_dir, new_dir)

def normalize_data(train_data, validation_data):
    # Normalize the images
    train_data = train_data.map(lambda x, y: (x / 255.0, y))
    validation_data = validation_data.map(lambda x, y: (x / 255.0, y))

    return train_data, validation_data