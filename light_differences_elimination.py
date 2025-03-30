
import numpy as np
import cv2
import global_variables

def light_diff_elimination(image1, image2_registered):
    import imageio
    from histogram_matching import ExactHistogramMatcher
    reference_histogram = ExactHistogramMatcher.get_histogram(image1)
    new_target_img = ExactHistogramMatcher.match_image_to_histogram(image2_registered, reference_histogram)
    cv2.imwrite(global_variables.output_dir + '/image2_registered_histogram_matched.jpg', new_target_img)
    new_target_img = np.asarray(new_target_img, dtype=np.uint8)
    return new_target_img