# Split image into smaller images of the numbers
# Turn these to tensors readable for MNIST

from PIL import Image
from typing import Tuple, List
from scipy import ndimage
import numpy as np
import cv2
import torch

def to_array(im: Image) -> np.ndarray:
    # Make the image grayscale
    im = im.convert('L')

    # Convert the image to an array
    array = np.array(im, dtype=np.float32)
    # Invert the image (MNIST is black background white text)
    array = 255 - array

    return array

def find_and_seperate_shapes(im_array: np.ndarray) -> Tuple[List[np.ndarray], 
                                                            List[Tuple[int, int, int, int]]]:
    # Binarise the image, so it can be used in ndimage
    binary_im = im_array > 5 # 5 so we have some leeway on what counts as a black pixel

    # Label the different shapes
    structure = np.ones((3, 3), dtype=int) # To count diagonal pixels as "connected"
    labelled_array, num_shapes = ndimage.label(binary_im, structure=structure)

    # Go through and cutout each shape
    shapes = []
    locations = []
    for label in range(1, num_shapes + 1):
        # Create a mask for the current label
        mask = labelled_array == label
        
        # Find the rows and columns where this shape exist
        rows = np.any(mask, axis=1)
        columns = np.any(mask, axis=0)
        # Find the indicies of these rows and columns
        row_inds = np.where(rows)[0]
        column_inds = np.where(columns)[0]
        # Find and save the binding box        
        r_st, r_end = row_inds[0], row_inds[-1]
        c_st, c_end = column_inds[0], column_inds[-1]
        locations.append((r_st, c_st, r_end, c_end))

        # Cutout out the shape
        shape = im_array[r_st:(r_end + 1), c_st:(c_end + 1)]
        shapes.append(shape)

    return shapes, locations

def format_for_MNIST(shapes: List[np.ndarray], mean, std) -> Tuple[List[np.ndarray]]:
    # MNIST wants a 28x28 image, however it also requires some padding,
    # so I will bring the image down to fit in 20x20, then centre it on
    # a 28x28 array

    new_shapes = []
    for shape in shapes:
        # Reduce image so largest side is 20 pixels long
        # Calculate new height and width
        h, w = shape.shape
        scale = 20 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)

        # change interpolation method depending on whether we are 
        # shrinking or enlarging the image
        interpol = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

        # Resize the array
        new_im = cv2.resize(shape, (nw, nh), interpolation=interpol)

        # Create the 28x28 image and centre the scaled shape on this 
        # (using centre of mass)
        new_shape = np.zeros((28, 28))
        center_r, center_c = ndimage.center_of_mass(new_im)
        # Round these values to the nearest int
        center_r, center_c = int(round(center_r)), int(round(center_c))
        # Calculate where to place the shape
        r_st, c_st = 14 - center_r, 14 - center_c
        r_end, c_end = r_st + nh - 1, c_st + nw - 1
        # Check we are in bounds
        r_st, c_st, r_end, c_end = max(0, r_st), max(0, c_st), min(28, r_end), min(28, c_end)
        # Form the new shape
        new_shape[r_st:(r_end + 1), c_st:(c_end + 1)] = new_im
        # Normalise the values
        new_shape = new_shape / 255
        new_shape = (new_shape - mean) / std
        # Save the shape
        new_shapes.append(new_shape)

    return new_shapes
    

def to_MNIST_tensor(im: Image) -> ...: # ADD THIS!
    im_array = to_array(im)

    shapes, locations = find_and_seperate_shapes(im_array)

    # Pull mean and std out of model
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("services/model/model_with_stats.pth", map_location=device)
    # Need to bring it out of tensor (.numpy()) then out of list ([0])
    mean = checkpoint["means"].numpy()[0]
    std = checkpoint["stds"].numpy()[0]

    # Format the shapes for MNIST
    shapes = format_for_MNIST(shapes, mean, std)

    # Combine the shapes into one numpy array and then turn it to a tensor
    all_shapes = np.stack(shapes, axis=0)
    tensor = torch.from_numpy(all_shapes)

    return tensor, locations