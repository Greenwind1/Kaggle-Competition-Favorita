# Image Normalizer

def img_normalize(img, mode=3):
    """ Normalize image array
    :param img: image array, batch array
    :param mode: int, 1: 0 ~ 1, 2: -1 ~ +1, 3: imagenet normalization,
    Others: return original image array
    """
    if mode == 1:
        img /= 255.
    elif mode == 2:
        img /= 127.5
        img -= 1.
    elif mode == 3:
        img /= 255.
        # Here it's ImageNet statistics
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Considering an ordering NCHW (batch, height, width, channel)
        for i in range(3):
            img[..., i] -= mean[i]
            img[..., i] /= std[i]
    else:
        return img
