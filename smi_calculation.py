from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageOps
from pathlib import Path
import shutil
import scipy.optimize


class Camera:

    def __init__(self, data_path):
        self.pi2 = np.pi / 2
        self.camera_rotation_angles = [-0.9* self.pi2, 0, 1.75 * self.pi2]
        self.data_path = data_path
        self.calib_path = self.data_path / 'calib'
        self.image_data_path = self.data_path / 'leftImage' / 'data'
        self.image_files = [x for x in self.image_data_path.glob('*.bmp') if x.is_file()]
        self.K, self.D = self.read_calib_data()

    def read_calib_data(self):
        cam_mono = cv2.FileStorage(str(self.calib_path / 'cam_mono.yml'), cv2.FILE_STORAGE_READ)
        K = cam_mono.getNode("K").mat()
        D = cam_mono.getNode("D").mat()
        K_final = np.array(K)
        D_final = np.array(D)
        K_final[0, 0] = 1.7 * K_final[0, 0]
        K_final[1, 1] = 1.7 * K_final[1, 1]
        D_final = [x[0] for x in D_final]
        return K_final, D_final

    def translation_pts_to_cam_sys(self, rotated_points, cam_point):
        cam_coord = [rotated_points[0] + cam_point[3],
                     rotated_points[1] + cam_point[4],
                     rotated_points[2] + cam_point[5]]
        return cam_coord

    def projection_pts_on_camera(self, translated_points):
        w = translated_points[2]
        if w.item() != 0:
            # translated_points[0] = translated_points[0] + 959.0
            # translated_points[1] = translated_points[1] + 539.0
            translated_points = translated_points / w
            pixel = np.dot(self.K, translated_points)
            if (abs(pixel[0]) < 960) and (abs(pixel[1]) < 540):
                # print(pixel)
                """
                r = translated_points[0] ** 2 + translated_points[1] ** 2
                Tan = math.atan(r)
                translated_points[0] = (1 + self.D[0] * r + self.D[1] * (r ** 2) + self.D[4] * (r ** 3)) * \
                                       translated_points[0] * Tan / r
                translated_points[1] = (1 + self.D[0] * r + self.D[1] * (r ** 2) + self.D[4] * (r ** 3)) * \
                                       translated_points[1] * Tan / r
                pixel = np.dot(self.K, translated_points)
                """
                return pixel

class Lidar:

    def __init__(self, data_path, camera):
        # lidar position in cam coord
        self.lidar_pos_in_cam_sys = [-0.885, -0.066, 0]

        self.data_path = data_path
        self.lidar_data_path = self.data_path / 'velodyne_points' / 'data'
        self.lidar_data = [x for x in self.lidar_data_path.glob('*.csv') if x.is_file()]
        self.local_cam = camera

    def projection_pts_on_cam(self, csv_data, cam_point=None, return_all=False):
        csv_data = csv_data[0:3]
        if cam_point is None:
            alpha = self.local_cam.camera_rotation_angles[0]
            betta = self.local_cam.camera_rotation_angles[1]
            gamma = self.local_cam.camera_rotation_angles[2]
            cam_pos = self.lidar_pos_in_cam_sys
        else:
            alpha = cam_point[0]
            betta = cam_point[1]
            gamma = cam_point[2]
            cam_pos = cam_point[3:]

        cam_point = [alpha, betta, gamma, cam_pos[0], cam_pos[1], cam_pos[2]]

        roll = [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]]
        pitch = [[np.cos(betta), 0, np.sin(betta)], [0, 1, 0],
                 [-np.sin(betta), 0, np.cos(betta)]]
        yaw = [[np.cos(gamma), -np.sin(gamma), 0],
               [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]
        r_matrix = np.dot(roll, np.dot(pitch, yaw))

        rotated_to_cam_points = np.dot(r_matrix, csv_data)
        #print("after rotation ", rotated_to_cam_points)

        translated_to_cam_pts = self.local_cam.translation_pts_to_cam_sys(rotated_to_cam_points, cam_point)
        #print("after translations", translated_to_cam_pts)

        point_height = translated_to_cam_pts[1]
        point_distance = np.sqrt(sum(i * i for i in translated_to_cam_pts))

        projected_to_camera_points = self.local_cam.projection_pts_on_camera(translated_to_cam_pts)
        #print("after projections", projected_to_camera_points)

        if return_all:
            return projected_to_camera_points, point_height, point_distance
        else:
            return projected_to_camera_points

    def dfScatter(self, mode, i, df, xcol='x', ycol='y', catcol='color', image_mod="ALL"):
        fig, ax = plt.subplots(figsize=(20, 10), dpi=60, )
        categories = np.unique(df[catcol])
        try:
            colors = np.linspace(categories.min(), categories.max(), len(categories))
        except ValueError:
            print("empty color dict")
            return None

        colordict = dict(zip(categories, colors))
        df["c"] = df[catcol].apply(lambda k: colordict[k])
        img = ImageOps.mirror(Image.open((self.local_cam.image_files[i])))
        sc = plt.scatter(df[xcol], df[ycol], c=df.c, zorder=1, s=10)

        if image_mod == "ALL":
            plt.imshow(img, extent=[df[xcol].min(), df[xcol].max(), df[ycol].min(), df[ycol].max()], zorder=0,
                   aspect='auto')

        colorize = plt.colorbar(sc, orientation="horizontal")
        if mode == "Height":
            colorize.set_label("Height (m)")
        else:
            colorize.set_label("Distance (m)")
        return fig


class SMI_calculations:
    def __init__(self, data_path):
        self.pi2 = np.pi / 2
        self.m = 1

        self.local_camera = Camera(data_path)
        self.local_lidar = Lidar(data_path, self.local_camera)
        self.image_path = Path(data_path) / 'images'
        if self.image_path.is_dir():
            shutil.rmtree(str(self.image_path), ignore_errors=True)
        self.image_path.mkdir()

    def power(self):

        average_points = [0, 0, 0, 0, 0, 0]
        for i in range(len(self.local_camera.image_files)):
            SMI_point = self.optimize(self.local_lidar.lidar_data[i], self.local_camera.image_files[i])
            for k in range(len(SMI_point)):
                average_points[k] += SMI_point[k]
                self.m += 1
            print("SMI point for", i, " file", SMI_point)
            
        SMI_point = [average_points[i] / self.m for i in range(len(average_points))]
        """
        SMI_point = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
        """

        self.SMI_Visualization(SMI_point, color="Distance")

    def optimize(self, Lidar_file, image):

        def func(x):
            print(x)
            SMI = 0
            list_intens = []
            list_ref = []
            for j in range(len(Lidar_data_np)):
                pixel = self.local_lidar.projection_pts_on_cam(Lidar_data_np[j,:3], x)
                if pixel is not None:
                    list_intens.append(self.get_intensivity(pixel, img))
                    list_ref.append(Lidar_data_np[j, 3])

            if list_ref is not None:
                kernel_r = gaussian_kde(list_ref)
                ref = kernel_r.evaluate(range(0, 255))
                kernel_i = gaussian_kde(list_intens)
                inte = kernel_i.evaluate(range(0, 255))
                mutual = np.histogram2d(list_ref, list_intens, bins=255, range=[[0, 255], [0, 255]], density=True)
                for i in range(0, 255):
                    for j in range(0, 255):
                        SMI += 0.5 * ref[i] * inte[j] * ((mutual[0][i][j] / (ref[i] * inte[j])) - 1) ** 2
            print("SMI ", -SMI)
            return -SMI

        Lidar_data_np = np.loadtxt(Lidar_file, delimiter=',')
        img = ImageOps.mirror(Image.open(image).convert('L'))

        SMI_0 = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
        result = scipy.optimize.minimize(func, x0=SMI_0, method= 'Nelder-Mead')
        return result.x

    def get_intensivity(self, pixel, img):
        x = int(pixel[0])
        y = int(pixel[1])
        return img.getpixel((x, y))

    def SMI_Visualization(self, SMI_point=None, color="Height"):
        if SMI_point is None:
            SMI_point = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys

        for i in range(len(self.local_lidar.lidar_data)):
            print("plot for", i, "file after calib,", "SMI_point =", SMI_point, ",color = ", color)

            Lidar_data_np = np.loadtxt(self.local_lidar.lidar_data[i], delimiter=',')

            x = []
            y = []
            heights = []
            distances = []

            for j in range(len(Lidar_data_np)):
                df = pd.DataFrame()
                pixel, height, distance = self.local_lidar.projection_pts_on_cam(Lidar_data_np[j,:3], return_all = True)
                if pixel is not None:
                    x.append(pixel[0])
                    y.append(pixel[1])
                    heights.append(height)
                    distances.append(distance)

            if color == "Height":
                df.insert(0, "x", x, True)
                df.insert(1, "y", y, True)
                df.insert(2, "color", heights, True)

            else:
                df.insert(0, "x", x, True)
                df.insert(1, "y", y, True)
                df.insert(2, "color", distances, True)

            fig = self.local_lidar.dfScatter(color, i, df,)
            fig2 = self.local_lidar.dfScatter(color, i, df, image_mod=None)
            if fig is not None:
                fig.savefig(str(self.image_path / str(i)) + '_at_SMI_point.png', dpi=60)
                fig2.savefig(str(self.image_path / str(i)) + '_at_SMI_img.png', dpi=60)