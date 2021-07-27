import glob
from os import path
import os
from PIL import Image
import random
import uuid
import pandas as pd


class DataPreparer:
    def __init__(self, element_count, data_path, data_output_path, leave_if_exist=True, photo_size=(1028, 1028)):

        self.element_count = element_count
        self.data_path = data_path
        self.data_output_path = data_output_path
        self.leave_if_exist = True
        self.photo_size = photo_size

        if not path.isdir(data_output_path):
            os.mkdir(data_output_path)

        if leave_if_exist and path.isfile(data_output_path + '/data.csv'):
            return

        if not leave_if_exist:
            os.remove(data_output_path + '/data.csv')
            for data_file in glob.glob(data_path + '/*.png'):
                os.remove(data_file)

        images = []

        for data_file in glob.glob(data_path + '/*.png'):
            images.append(Image.open(data_file))

        img_size = len(images) - 1

        background_width = photo_size[0]
        background_height = photo_size[1]

        file_name = []
        x_cord_list = []
        y_cord_list = []
        width_list = []
        height_list = []

        for count in range(element_count):
            value = random.randint(0, img_size)
            img_buff = images[value]
            width, height = img_buff.size

            number = random.randint(400, 600)
            width += number - random.randint(0, 20)
            height += number

            img_buff = img_buff.resize((width, height))
            file_uuid = uuid.uuid4().__str__()

            width, height = img_buff.size

            x_value = random.randint(0, background_width - width)
            y_value = random.randint(0, background_height - height)

            background_image = Image.new(mode='RGB', size=photo_size, color=(0, 0, 0))
            background_image.paste(im=img_buff,
                                   box=(x_value,
                                        y_value))
            path_for_csv = file_uuid + '.png'
            background_image.save(fp=data_output_path + '/' + path_for_csv)

            file_name.append(path_for_csv)
            x_cord_list.append(x_value)
            y_cord_list.append(y_value)
            width_list.append(width)
            height_list.append(height)

        data_frame = pd.DataFrame({
            'x_cord': x_cord_list,
            'y_cord': y_cord_list,
            'width': width_list,
            'height': height_list,
            'file_name': file_name
        })

        data_frame.to_csv(data_output_path + '/data.csv', index=False)