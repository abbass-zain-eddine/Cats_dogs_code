def gamma_correction(image, gamma):
    """
    Apply gamma correction to the input image.

    Parameters:
    image (numpy.ndarray): Input image to be corrected.
    gamma (float): Gamma value for correction. Values less than 1 will brighten the image, while values greater than 1 will darken it.

    Returns:
    numpy.ndarray: Gamma corrected image.
    """
    # Create a lookup table for gamma correction
    inv_gamma = 1.0 / gamma
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    
    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, np.array(table, dtype=np.uint8))