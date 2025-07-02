import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm

ROOT = '/Users/yezi/Documents/torchCode/data_eyes/'


def double_csv(input_file, output_file):
    """
    original label file: "name, label" for each sample
    new label file: "name, label" (1st row)
                    "name_p, label" (2nd row) for each sample
    """
    df = pd.read_csv(input_file)
    new_rows = []
    # for _, row in df.iterrows():
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # Keep the original row as is.
        new_rows.append(row)

        # copy and rename
        new_row = row.copy()
        # new_row['id_code'] = f"{row['id_code'].rsplit('.', 1)[0]}_p"
        new_row['id_code'] = f"{row['id_code']}_p"
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_file, index=False)
    print(f"done！new csv has been saved as {output_file}")


class Preprocess2views(object):
    """
    preprocessing each image using "self.__preprocess_ori_img(...)" and "self.__clahe(...)",
    and then saving the preprocessed images as "name" and "name_p" in a NEW dir
    """
    def __init__(self, img_path, target_path):
        self.dir = img_path
        self.len = 0
        self.scale = 300
        self.target_path = target_path

    def generate2views(self):
        # getting all available image names
        img_files = [
            f for f in os.listdir(self.dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.len = 0

        for img_name in tqdm(img_files, desc="Processing images"):
            img_path = os.path.join(self.dir, img_name)
            img = cv2.imread(img_path, 1)

            if img is None:
                print(f'Failed to read: {img_name}, possibly broken image')
                continue

            # Remove black borders first
            img = self.__remove_black_borders(img)
            self.len += 1

            # Save the 1st processed version with '_p' suffix
            img_processed = self.__preprocess_ori_img(img)
            name, ext = os.path.splitext(img_name)
            img_name_p = "{}_p{}".format(name, ext)  # e.g., aaa.jpg-->aaa_p.jpg
            cv2.imwrite(os.path.join(self.target_path, img_name_p), img_processed)

            # Ssvinh the 2nd processed version (CLAHE) as the original file name
            img_processed2 = self.__clahe(img)
            cv2.imwrite(os.path.join(self.target_path, img_name), img_processed2)

        print('successfully preprocessing {} images'.format(self.len))

    @staticmethod
    def __remove_black_borders(img):
        """Remove black borders from the image"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Threshold the image to find non-black regions
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        # Find contours of non-black regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return img

        # Get the bounding box of the largest contour
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        margin = 5
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, img.shape[1] - x)
        h = min(h + 2 * margin, img.shape[0] - y)

        # Crop the image to the bounding box
        cropped = img[y:y + h, x:x + w]

        return cropped

    def __preprocess_ori_img(self, img):
        # scaling
        img_processed = self.__scaleRadius(img, self.scale)
        # substracting local avg
        img_processed = cv2.addWeighted(img_processed, 4,
                                        cv2.GaussianBlur(img_processed, (0, 0), self.scale / 30),
                                        -4, 128)
        return img_processed

    def __scaleRadius(self, img, scale):
        x = img[img.shape[0] // 2, :, :].sum(1)
        r = (x > x.mean() / 10).sum() / 2 + 1e-6
        s = scale * 1.0 / r
        return cv2.resize(img, (0, 0), fx=s, fy=s)

    def __clahe(self, img):
        imgb = img[:, :, 0]
        imgg = img[:, :, 1]
        imgr = img[:, :, 2]

        claheb = cv2.createCLAHE(clipLimit=1, tileGridSize=(10, 18))
        claheg = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 18))
        claher = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 18))
        cllb = claheb.apply(imgb)
        cllg = claheg.apply(imgg)
        cllr = claher.apply(imgr)

        return np.dstack((cllb, cllg, cllr))

def main(split):
    print("preprocessing {} images".format(split))
    input_csv = os.path.join(ROOT, 'aptos2019/{}.csv'.format(split))  # your GT file
    output_csv = os.path.join(ROOT, 'aptos2019/{}2view.csv'.format(split))  # the new GT file to be generated
    image_path = os.path.join(ROOT, 'aptos2019/{}_images'.format(split))  # your image path
    target_path = os.path.join(ROOT, 'aptos2019/{}_preprocessed'.format(split))  # path for saving preprocessed images
    # generating new GT file
    double_csv(input_file=input_csv, output_file=output_csv)
    # generating new image path
    os.makedirs(target_path, exist_ok=True)
    # preprocessing and saving
    preprocess_tool = Preprocess2views(img_path=image_path,
                                       target_path=target_path)
    preprocess_tool.generate2views()


if __name__ == "__main__":
    # main(split='train')
    main(split='test')