import keras.backend as K
## intersection over union
def IoU(y_true, y_pred, eps=1e-6):
    # if np.max(y_true) == 0.0:
    #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)

def iou_loss(in_gt, in_pred):
    return - IoU(in_gt, in_pred)