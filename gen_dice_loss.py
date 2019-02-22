import keras.backend as K

def generalized_dice_coeff(y_true, y_pred):
    '''
    https://arxiv.org/pdf/1707.03237.pdf
    '''
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))  # Count the number of pixels in the target area
    w = 1/(w**2+0.000001) # Calculate category weights
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)  #molecular


    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)  #denominator

    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)
