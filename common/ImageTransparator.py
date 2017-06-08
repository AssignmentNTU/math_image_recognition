import numpy as np
from PIL import Image
from skimage import io


class ImageTransparater:

    threshold = 100
    dist = 5

    TRESHOLD_AS_BLACK = 10

    def __init__(self):
         return

    def start_transforming(self, file_name, return_image=False):
        if not return_image:
            img = Image.open(file_name).convert('RGBA')
            # np.asarray(img) is read only. Wrap it in np.array to make it modifiable.
            arr = np.array(np.asarray(img))
            r, g, b, a = np.rollaxis(arr, axis=-1)
            mask = ((r > self.threshold)
                    & (g > self.threshold)
                    & (b > self.threshold)
                    # & (np.abs(r - g) < dist)
                    # & (np.abs(r - b) < dist)
                    # & (np.abs(g - b) < dist)
                    )
            arr[mask, 3] = 0
            img = Image.fromarray(arr, mode='RGBA')
            out_image_name = self.remove_any_dot_extension(file_name)+"_out_transparent.png"
            img.save(out_image_name)
            img.close()
            return out_image_name

        else:
            # if return image is true then the file_name will be image object
            arr = np.array(np.asarray(file_name))
            r, g, b, a = np.rollaxis(arr, axis=-1)
            mask = ((r > self.threshold)
                    & (g > self.threshold)
                    & (b > self.threshold)
                    # & (np.abs(r - g) < dist)
                    # & (np.abs(r - b) < dist)
                    # & (np.abs(g - b) < dist)
                    )
            arr[mask, 3] = 0
            img = Image.fromarray(arr, mode='RGBA')
            return img

    def remove_any_dot_extension(self, full_file_name):

        file_name_without_extension = full_file_name.split(".")[0]
        return file_name_without_extension

    def start_transforming_back_to_jpg(self, filename, return_array=False):

        image = io.imread(filename)

        for w in range(len(image)):
            for h in range(len(image[w])):
                curr_pixel_color = image[w, h]

                if curr_pixel_color <= self.TRESHOLD_AS_BLACK:
                    image[w, h] = 0
                else:
                    image[w, h] = 255

        image_name = self.remove_any_dot_extension(filename) + "_white.png"
        io.imsave(image_name, image)

        return image_name if not return_array else image
