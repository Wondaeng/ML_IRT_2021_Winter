import cv2
from detectron2.engine import DefaultPredictor


def uncertainty_score(base_path, pool, cfg, mode='min_confidence'):
    predictor = DefaultPredictor(cfg)
    confidence_list = []
    for f in pool:
        image_path = base_path + f
        img = cv2.imread(image_path)
        output = predictor(img)
        num_instances = len(output['instances'])

        if mode == 'min_confidence':  # Criterion: The image contains the least confident instance
            if num_instances > 0:
                confidence_score = 1  # 100% confident
                for i in range(num_instances):
                    confidence = output['instances'].scores[i].item()
                    if confidence <= confidence_score:
                        confidence_score = confidence
                confidence_list.append((confidence_score, f))
            else:
                confidence_list.append((0, f))  # False Negative

        elif mode == 'mean_confidence':    # Criterion: The image with the least mean confident
            if num_instances > 0:
                confidence_score = 0
                for i in range(num_instances):
                    confidence = output['instances'].scores[i].item()
                    confidence_score += confidence
                confidence_mean = confidence_score / num_instances
                confidence_list.append((confidence_mean, f))
            else:
                confidence_list.append((1, f))

    return confidence_list


