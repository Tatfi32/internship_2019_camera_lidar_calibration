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
        self.camera_rotation_angles = [ -1 * self.pi2, 0, 2 * self.pi2]
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
        K_final[0, 0] = 2 * K_final[0, 0]
        K_final[1, 1] = 2 * K_final[1, 1]
        D_final = [x[0] for x in D_final]
        return K_final, D_final

    def projection_pts_on_camera(self, pts_in_cam_sys):
        """
        Args:
            pts_in_cam_sys:lidar points in camera coordinates system to be projected on camera matrix

        Returns:
            coordinates of projected to matrix points (pixel)
        """
        w = pts_in_cam_sys[2]
        if w.item() != 0:
            pts_in_cam_sys = pts_in_cam_sys / w
            pixel = np.dot(self.K, pts_in_cam_sys)
            if (abs(pixel[0]) < 960) and (abs(pixel[1]) < 540):
                """
                r = pts_in_cam_sys[0] ** 2 + pts_in_cam_sys[1] ** 2
                Tan = math.atan(r)
                pts_in_cam_sys[0] = (1 + self.D[0] * r + self.D[1] * (r ** 2) + self.D[4] * (r ** 3)) * \
                                    pts_in_cam_sys[0] * Tan / r
                pts_in_cam_sys[1] = (1 + self.D[0] * r + self.D[1] * (r ** 2) + self.D[4] * (r ** 3)) * \
                                    pts_in_cam_sys[1] * Tan / r
                pixel = np.dot(self.K, pts_in_cam_sys)
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

    def rotation_pts_to_cam_sys(self, alpha, betta, gamma, lidar_point):
        """
        Args:
            alpha: angle for roll matrix
            betta: angle for pitch pitch
            gamma: angle for yaw yaw
            lidar_point: points in lidar coordinates to be rotated to camera system

        Returns:
            rotated to camera system lidar point
        """
        roll = [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]]
        pitch = [[np.cos(betta), 0, np.sin(betta)], [0, 1, 0],
                 [-np.sin(betta), 0, np.cos(betta)]]
        yaw = [[np.cos(gamma), -np.sin(gamma), 0],
               [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]
        r_matrix = np.dot(roll, np.dot(pitch, yaw))

        rotated_to_cam_points = np.dot(r_matrix, lidar_point)
        return rotated_to_cam_points

    def translation_pts_to_cam_sys(self, rotated_points, translation_vector):
        """
        Args:
            rotated_points:already rotated to camera coordinates position of lidar point to be translater in lidar coordinates
            translation_vector: translation vector (lidar position in camera coordinates)
        Returns:
            lidar points in camera coordinates

        """
        #print("rotated_points",rotated_points)
        #print("translation_vector",translation_vector )
        translated_to_cam_pts = [rotated_points[0] + translation_vector[0],
                                 rotated_points[1] + translation_vector[1],
                                 rotated_points[2] + translation_vector[2]]
        return translated_to_cam_pts

    def projection_pts_on_cam(self, lidar_point, SMI_point, return_all=False):
        """
        Args:
            lidar_point: point coordinats in lidar system
            SMI_point: extrinsic parameters rotation and translation (alpha, betta, gamma, x, y, z)
            return_all:
               True: mode for point distances and heights calculations during projection

        Returns:
            projected to camera matrix point coordinates

        """
        #print("lidar_point",lidar_point)
        #print("SMI_point", SMI_point)
        alpha = SMI_point[0]
        betta = SMI_point[1]
        gamma = SMI_point[2]
        translation_vector = SMI_point[3:]

        rotated_to_cam_points = self.rotation_pts_to_cam_sys(alpha, betta, gamma, lidar_point)
        #print("after rotation ", rotated_to_cam_points)

        pts_in_cam_sys = self.translation_pts_to_cam_sys(rotated_to_cam_points, translation_vector)
        #print("after translations", pts_in_cam_sys)

        if return_all:
            point_height = pts_in_cam_sys[1]
            point_distance = np.linalg.norm (pts_in_cam_sys, axis=0)
            projected_to_camera_points = self.local_cam.projection_pts_on_camera(pts_in_cam_sys)
            # print("after projections", projected_to_camera_points)
            return projected_to_camera_points, point_height, point_distance

        else:
            projected_to_camera_points = self.local_cam.projection_pts_on_camera(pts_in_cam_sys)
            #print("after projections", projected_to_camera_points)
            return projected_to_camera_points


class SMI_calculations:
    def __init__(self, data_path):

        self.m = 0
        self.points = []

        self.local_camera = Camera(data_path)
        self.local_lidar = Lidar(data_path, self.local_camera)

        self.image_path = Path(data_path) / 'images'
        if self.image_path.is_dir():
            shutil.rmtree(str(self.image_path), ignore_errors=True)
        self.image_path.mkdir()

        self.local_visual = Visualization(self.local_lidar, self.local_camera)

    def power(self):
        """main tread"""

        """ Points plot before calibration"""
        self.local_visual.SMI_Visualization(SMI_point=None, color="Distance", files_num=10)

        """Show SMI values near ground_true point - all parameters change independently"""
        #self.plot_near_ground_point(files_num=2)

        """Calculate average optimal point for N "files_num" and 
        plot SMI values near optimal point (with simultaneous parameters changes)"""
        SMI_point = self.average_SMI_point(files_num=10)

        """ Points plot after calibration"""
        self.local_visual.SMI_Visualization(SMI_point, color="Distance", files_num=10)

    def plot_near_ground_point(self,files_num):

        if files_num is None:
            files_num = len(self.local_camera.image_files)

        points = np.zeros((7, 21))  # range for changing start parameters
        for i in range(0, 21):
            points[6][i] = float(i - 10) / 200  # number of points in changing interval

        for file in range(files_num):
            print("Getting SMI values near ground_true point for", file, "file")

            for i in range(0, 21):
                SMI_0_alpha = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
                SMI_0_alpha[0] = SMI_0_alpha[0] - points[6][i]
                points[0][i] = self.optimize(self.local_lidar.lidar_data[i], self.local_camera.image_files[i], SMI_0_alpha, mode="plot")
            plt.subplot(321)
            plt.plot(points[6][:], points[0][:])
            plt.ylabel('SMI ground_true alpha')
            plt.xlabel('indent from ground_point, m')


            for i in range(0, 21):
                SMI_0_betta = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
                SMI_0_betta[1] = SMI_0_betta[1] - points[6][i]
                points[1][i] = self.optimize(self.local_lidar.lidar_data[i], self.local_camera.image_files[i], SMI_0_betta, mode="plot")
            plt.subplot(323)
            plt.plot(points[6][:], points[1][:])
            plt.ylabel('SMI ground_true betta')
            plt.xlabel('indent from ground_point, m')

            for i in range(0, 21):
                SMI_0_gamma = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
                SMI_0_gamma[2] = SMI_0_gamma[2] - points[6][i]
                points[2][i] = self.optimize(self.local_lidar.lidar_data[i], self.local_camera.image_files[i], SMI_0_gamma, mode="plot")
            plt.subplot(325)
            plt.plot(points[6][:], points[2][:])
            plt.ylabel('SMI ground_true gamma')
            plt.xlabel('indent from ground_point, m')

            for i in range(0, 21):
                SMI_0_x = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
                SMI_0_x[3] = SMI_0_x[3] - points[6][i]
                points[3][i] = self.optimize(self.local_lidar.lidar_data[i], self.local_camera.image_files[i], SMI_0_x, mode="plot")
            plt.subplot(322)
            plt.plot(points[6][:], points[3][:])
            plt.ylabel('SMI ground_true x')
            plt.xlabel('indent from ground_point, m')

            for i in range(0, 21):
                SMI_0_y = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
                SMI_0_y[4] = SMI_0_y[4] - points[6][i]
                points[4][i] = self.optimize(self.local_lidar.lidar_data[i], self.local_camera.image_files[i], SMI_0_y, mode="plot")
            plt.subplot(324)
            plt.plot(points[6][:], points[4][:])
            plt.ylabel('SMI ground_true y')
            plt.xlabel('indent from ground_point, m')

            for i in range(0, 21):
                SMI_0_z = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
                SMI_0_z[5] = SMI_0_z[5] - points[6][i]
                points[5][i] = self.optimize(self.local_lidar.lidar_data[i], self.local_camera.image_files[i], SMI_0_z,
                                             mode="plot")
            plt.subplot(326)
            plt.plot(points[6][:], points[5][:])
            plt.ylabel('SMI ground_true z')
            plt.xlabel('indent from ground_point, m')

            plt.show()

    def plot_near_optimal_point(self,):
        """
        Create 6 plots of process for extrinsic parameters finding with scipy.optimize.minimize
        All extrinsic parameters change simultaneously at the same time

        Each of 6 plots represent SMI value and relevant extrinsic parameter value
        (alpha,betta, gamma) - rotation, each inside (roll, pitch, yaw) matrices
        (x,y,z) - translation (x, y: along camera's matrix axes, z: towards to camera )
        """
        points_np = [np.fromiter(k.values(), dtype="float") for k in self.points]
        points_np = np.array(points_np)

        plt.subplot(321)
        plt.scatter(points_np[:,0], points_np[:,6])
        plt.ylabel('SMI for alpha')
        plt.xlabel('optimal alpha value, m')

        plt.subplot(323)
        plt.scatter(points_np[:, 1], points_np[:, 6])
        plt.ylabel('SMI for betta')
        plt.xlabel('optimal betta value, m')

        plt.subplot(325)
        plt.scatter(points_np[:, 2], points_np[:, 6])
        plt.ylabel('SMI for gamma')
        plt.xlabel('optimal gamma value, m')

        plt.subplot(322)
        plt.scatter(points_np[:, 3], points_np[:, 6])
        plt.ylabel('SMI for x')
        plt.xlabel('optimal x value, m')

        plt.subplot(324)
        plt.scatter(points_np[:, 4], points_np[:, 6])
        plt.ylabel('SMI for y')
        plt.xlabel('optimal y value, m')

        plt.subplot(326)
        plt.scatter(points_np[:, 5], points_np[:, 6])
        plt.ylabel('SMI for z')
        plt.xlabel('optimal z value, m')
        plt.show()
        self.points = []

    def average_SMI_point(self, files_num):
        """
        Args:
            files_num: number of image files to be processed for average SMI_point value calculation

        Returns:
            SMI_point:  average for files_num extrinsic parameters value- 6 number (alpha,betta,gamma, x, y, z)
            (x,y,z) - translation (x, y: along camera's matrix axes, z: towards to camera )
            (alpha,betta, gamma) - rotation, each inside (roll, pitch, yaw) matrices
        """
        average_points = [0, 0, 0, 0, 0, 0]
        if files_num is None:
            files_num = len(self.local_camera.image_files)

        for i in range(files_num):
            SMI_point = self.optimize(self.local_lidar.lidar_data[i], self.local_camera.image_files[i])

            print("SMI point for", i, " file", SMI_point)
            self.m += 1
            for k in range(len(SMI_point)):
                average_points[k] += SMI_point[k]

        SMI_point = [(average_points[i] / self.m) for i in range(len(average_points))]
        print("Average SMI_point", SMI_point , "for", files_num, "files")
        return SMI_point

    def optimize(self, Lidar_file, image, SMI_0=None, mode="minimize"):
        """
        Args:
            Lidar_file: *csv data to be load and processed
            image: *bmp data to be load and processed
            SMI_0: initial extrinsic parameters value - 6 numbers (alpha,betta, gamma,x,y,z)

        Returns:
            result.x: optimal extrinsic parameters value according scipy.optimize.minimize
            #result.func: SMI value in optimal point
            #self.points.append only used for self.plot_near_optimal_point to show the working process
        """

        def func(x):
            SMI = 0
            list_intens = []
            list_ref = []
            for j in range(len(Lidar_data_np)):
                pixel = self.local_lidar.projection_pts_on_cam(Lidar_data_np[j, :3], x)
                if pixel is not None:
                    list_intens.append(self.get_intensivity(pixel, img))
                    list_ref.append(int(Lidar_data_np[j, 3]))
            if (list_ref != []) and (list_intens != []):
                kernel_r = gaussian_kde(list_ref)
                ref = kernel_r.evaluate(range(0, 255))
                kernel_i = gaussian_kde(list_intens)
                inte = kernel_i.evaluate(range(0, 255))
                mutual = np.histogram2d(list_ref, list_intens, bins=255, range=[[0, 255], [0, 255]], density=True)
                for i in range(0, 255):
                    for j in range(0, 255):
                        SMI += 0.5 * ref[i] * inte[j] * ((mutual[0][i][j] / (ref[i] * inte[j])) - 1) ** 2
                SMI += khi * len(list_ref)

                self.points.append({"alpha": x[0],"betta": x[1], "gamma": x[2],"x": x[3], "y": x[4], "z": x[5], "val":-SMI})
                return -SMI
            else:
                return 0

        khi = 0.1
        Lidar_data_np = np.loadtxt(Lidar_file, delimiter=',')
        img = ImageOps.mirror(Image.open(image).convert('L'))

        if SMI_0 is None:
            SMI_0 = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys

        if mode == "minimize":
            print("Calculate average optimal point")
            result = scipy.optimize.minimize(func, x0=SMI_0, method = 'Nelder-Mead',  options={'maxiter': 1})
            self.plot_near_optimal_point()
            return result.x

        if mode == "plot":
            return func(SMI_0)

    def get_intensivity(self, pixel, img):
        x = int(pixel[0])
        y = int(pixel[1])
        return img.getpixel((x, y))


class Visualization:

    def __init__(self, lidar, camera):
        self.local_lidar = lidar
        self.local_camera = camera
        self.image_path = self.local_lidar.data_path / 'images'


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
        img = ImageOps.mirror(Image.open((self.local_camera.image_files[i])))
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

    def SMI_Visualization(self, SMI_point=None, color="Height", files_num = None):
        """
        Args:
            SMI_point: value of extrinsic parameters (alpha,betta, gamma,x,y,z) to image process projection
            color: ="Height":plot color value as height values of each point
                   else: plot color value as distance values of each point
            files_num: number of *csv to be projected on relevant *bmp files

        Returns:
            create "images" folder in "data_path" directory and
            save images of projected *csv data on camera matrix with given SMI_point value
        """

        mode = "after"
        if SMI_point is None:
            SMI_point = self.local_camera.camera_rotation_angles + self.local_lidar.lidar_pos_in_cam_sys
            mode = "before"
        if files_num is None:
            files_num = len(self.local_camera.image_files)

        for i in range(files_num):
            print("plot", mode,"calib for", i, "file,", "SMI_point =", SMI_point, ", color = ", color)
            Lidar_data_np = np.loadtxt(self.local_lidar.lidar_data[i], delimiter=',')
            x = []
            y = []
            heights = []
            distances = []

            for j in range(len(Lidar_data_np)):
                df = pd.DataFrame()
                pixel, height, distance = self.local_lidar.projection_pts_on_cam(Lidar_data_np[j, :3], SMI_point, return_all=True)
                if pixel is not None:
                    x.append(pixel[0])
                    y.append(pixel[1])
                    if color == "Height":
                        heights.append(height)
                    else:
                        distances.append(distance)

            df.insert(0, "x", x, True)
            df.insert(1, "y", y, True)
            if color == "Height":
                df.insert(2, "color", heights, True)
            else:
                df.insert(2, "color", distances, True)


            fig  = self.dfScatter(color, i, df, )
            fig2 = self.dfScatter(color, i, df, image_mod=None)

            if fig is not None:
                if mode=="before":
                    fig.savefig(str(self.image_path / str(i)) + '_at_SMI_point_before.png', dpi=60)
                    fig2.savefig(str(self.image_path / str(i)) + '_at_SMI_img_before.png', dpi=60)
                    plt.close(fig)
                    plt.close(fig2)
                else:
                    fig.savefig(str(self.image_path / str(i)) + '_at_SMI_point_after.png', dpi=60)
                    fig2.savefig(str(self.image_path / str(i)) + '_at_SMI_img_after.png', dpi=60)
                    plt.close(fig)
                    plt.close(fig2)
