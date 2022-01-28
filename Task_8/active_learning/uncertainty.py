import cv2
import math
from detectron2.engine import DefaultPredictor


def uncertainty_score(base_path, pool, cfg, mode='mean_entropy'):
    predictor = DefaultPredictor(cfg)
    entropy_list = []
    for f in pool:
        image_path = base_path + f
        img = cv2.imread(image_path)
        output = predictor(img)
        num_instances = len(output['instances'])

        if num_instances > 0:
            entropy_sum = 0
            for i in range(num_instances):
                confidence = output['instances'].scores[i].item()
                score_entropy = (-1) * (confidence) * math.log10(confidence) + (-1) * (1 - confidence) * math.log10(1 - confidence)
                entropy_sum += score_entropy
            entropy_mean = entropy_sum/num_instances
            entropy_list.append((entropy_mean, f))
        else:
            entropy_list.append((0, f))
    return entropy_list



