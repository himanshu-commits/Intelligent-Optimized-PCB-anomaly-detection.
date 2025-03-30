import argparse
import cv2
import numpy as np
from numpy import loadtxt
from pathlib import Path

def main(results_dir):
    clustering_map = loadtxt(results_dir+'/clustering_data.csv', delimiter=',')
    accepted_classes = loadtxt(results_dir + '/accepted_classes.csv', delimiter=',')
    gt = cv2.imread(results_dir + "/GT.jpg")
    gt = cv2.resize(gt, (clustering_map.shape[1],clustering_map.shape[0] ), interpolation=cv2.INTER_AREA)
    recall = 0
    precision = 0
    gt_size = 0
    selected_size = 0
    print("Unique GT values:", np.unique(gt.reshape(-1, 3), axis=0))  # Check all unique colors
    print("Unique Clustering Map Values:", np.unique(clustering_map))
    print("Accepted Classes:", accepted_classes)
    print("GT Cluster Values:", np.unique(clustering_map[gt[:,:,0] >= 200]))
    
    
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if np.all(gt[i, j] >= [120, 120, 120]):  # Adjust threshold
                gt_size += 1
                if clustering_map[i,j] in accepted_classes:
                    recall += 1
            if clustering_map[i,j] in accepted_classes:
                selected_size+=1
                if np.all(gt[i, j] >= [120, 120, 120]):  # Adjust threshold
                    precision += 1
                    
    
    recall = recall / gt_size
    precision= precision / selected_size
    print("Recall", round(recall,4))
    print("Precision", round(precision,4))
    return recall, precision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for Running')
    parser.add_argument('-results_dir',
                        dest='results_dir',
                        help='destination of the results to evaluate')
    args = parser.parse_args()
    main(args.results_dir)
    
