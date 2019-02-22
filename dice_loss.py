import keras.backend as K

def dice_coef(y_true, y_pred, smooth=1):
    # if np.max(y_true) == 0.0:                                                  
    #     return dice_coef(1-y_true, 1-y_pred) ##if need for empty image; 
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return - dice_coef(in_gt, in_pred)


    