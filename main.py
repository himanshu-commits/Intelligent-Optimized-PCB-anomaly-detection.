import sys
sys.path.append('/home/yonif/.conda/envs/pca_kmeans_change_detection/lib/python3.6/site-packages')
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import cv2
import time
from PCA_Kmeans import compute_change_map, find_group_of_accepted_classes_DBSCAN, draw_combination_on_transparent_input_image
import global_variables
import os
import imageio.v2 as imageio
import argparse


def main(output_dir,input_path,reference_path,n,window_size, pca_dim_gray, pca_dim_rgb,
        lighting_fix, use_homography, resize_factor, save_extra_stuff):
    '''
 output_dir: destination directory for the output
 input_path: path to the Defected image
 reference_path: path to the Golden image
 n: number of classes for clustering the diff descriptors
 window_size: window size for the diff descriptors
 pca_dim_gray: pca target dimension for the gray diff descriptor
 pca_dim_rgb: pca target dimension for the rgb diff descriptor
 lighting_fix: true to enable histogram matching
 use_homography: true to enable SIFT homography (always recommended)
 resize_factor: scale the input images, usually with factor smaller than 1 for faster results
 save_extra_stuff: save diagnostics and extra results, usually for debugging
 return: the results are saved in output_dir
    '''
    global_variables.init(output_dir, save_extra_stuff) #setting global variables

    if use_homography:
        from registration import homography
    if lighting_fix:
        from light_differences_elimination import light_diff_elimination


    #for time estimations
    start_time = time.time()

    #read the inputs
    image_1 = imageio.imread(input_path)
    image_2 = imageio.imread(reference_path)
    print(image_1.shape)  # Should output (height, width, 3)
    print(image_2.shape)  # Should output (height, width, 3)



    #we need the images to be the same size. resize_factor is for increasing or decreasing further the images
    new_shape = (int(resize_factor*0.5*(image_1.shape[1]+image_2.shape[1])), int(resize_factor*0.5*(image_1.shape[0]+image_2.shape[0])))
    image_1 = cv2.resize(image_1,new_shape, interpolation=cv2.INTER_AREA)
    image_2 = cv2.resize(image_2, new_shape, interpolation=cv2.INTER_AREA)
    global_variables.set_size(new_shape[0],new_shape[1])
    
    if use_homography:
        image2_registered, blank_pixels = homography(image_1, image_2)
    else:
        image2_registered  = image_2

    if use_homography:
        image_1[blank_pixels] = [0,0,0]
        image2_registered[blank_pixels] = [0, 0, 0]
        print(image_1.shape)  # Should output (height, width, 3)
        print(image2_registered.shape)  # Should also output (height, width, 3)


    if (global_variables.save_extra_stuff):
        cv2.imwrite(global_variables.output_dir+ '/resized_blanked_1.jpg', image_1)

    if (lighting_fix):
        #Using the histogram matching, only image2_registered is changed
        image2_registered = light_diff_elimination(image_1, image2_registered)

        print("--- Preprocessing time - %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    clustering_map, mse_array, size_array = compute_change_map(image_1, image2_registered, window_size=window_size,
                                                               clusters=n, pca_dim_gray= pca_dim_gray, pca_dim_rgb=pca_dim_rgb)

    clustering = [[] for _ in range(n)]
    for i in range(clustering_map.shape[0]):
        for j in range(clustering_map.shape[1]):
            clustering[int(clustering_map[i,j])].append([i,j])

    input_image = imageio.imread(input_path)
    input_image = cv2.resize(input_image,new_shape, interpolation=cv2.INTER_AREA)
    b_channel, g_channel, r_channel = cv2.split(input_image)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    alpha_channel[:, :] = 50
    groups = find_group_of_accepted_classes_DBSCAN(mse_array)
    for group in groups:
        transparent_input_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        result = draw_combination_on_transparent_input_image(mse_array, clustering, group, transparent_input_image)
        cv2.imwrite(global_variables.output_dir + '/ACCEPTED_CLASSES'+'.png', result)

    print("--- PCA-Kmeans + Post-processing time - %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for Running')
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        help='destination directory for the output')
    parser.add_argument('--input_path',
                        dest='input_path',
                        help='path to the input image')
    parser.add_argument('--reference_path',
                        dest='reference_path',
                        help='path to the reference image')
    parser.add_argument('--n',
                        dest='n',
                        help='number of classes for clustering the diff descriptors')
    parser.add_argument('--window_size',
                        dest='window_size',
                        help='window size for the diff descriptors')
    parser.add_argument('--pca_dim_gray',
                        dest='pca_dim_gray',
                        help='pca target dimension for the gray diff descriptor')
    parser.add_argument('--pca_dim_rgb',
                        dest='pca_dim_rgb',
                        help='pca target dimension for the rgb diff descriptor')
    parser.add_argument('--pca_target_dim',
                        dest='pca_target_dim',
                        help='pca target dimension for final combination of the descriptors')
    parser.add_argument('--lighting_fix',
                        dest='lighting_fix',
                        help='true to enable histogram matching',
                        default=False, action='store_true')
    parser.add_argument('--use_homography',
                        dest='use_homography',
                        help='true to enable SIFT homography (always recommended)',
                        default=False, action='store_true')
    parser.add_argument('--resize_factor',
                        dest='resize_factor',
                        help='scale the input images, usually with factor smaller than 1 for faster results')
    parser.add_argument('--save_extra_stuff',
                        dest='save_extra_stuff',
                        help='save diagnostics and extra results, usually for debugging',
                        default=False, action='store_true')
    if __name__ == '__main__':
        args = parser.parse_args()
        main(args.output_dir, args.input_path, args.reference_path, int(args.n), int(args.window_size),  
            int(args.pca_dim_gray), int(args.pca_dim_rgb), bool(args.lighting_fix), bool(args.use_homography),
            float(args.resize_factor), bool(args.save_extra_stuff))
    