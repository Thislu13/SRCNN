import cv2
import os


path = r'/home/lu13/Documents/BSDS300/images'

size_count = {}

for dir in os.listdir(path):
    path_dir = os.path.join(path, dir)
    index = 1
    for imgname in os.listdir(path_dir):
        path_img = os.path.join(path_dir, imgname)
        img = cv2.imread(path_img)
        _h, _w = img.shape[:2]
        # print(path_img)
        # print(_w, _h)
        w_h = str(_w) + '*' + str(_h)
        # print(w_h)
        if w_h not in size_count:
            size_count[w_h] = 1
        else:
            size_count[w_h] += 1
        new_path = f'{index}.jpg'
        new_path = os.path.join(path_dir, new_path)
        print(new_path)
        index += 1
        os.rename(path_img, new_path)

print(size_count)