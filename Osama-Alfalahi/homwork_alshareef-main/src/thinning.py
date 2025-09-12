import numpy

def ZSThinningIteration(originalImageArray, iteration):
    I = originalImageArray.copy()
    M = numpy.zeros(originalImageArray.shape, numpy.uint8)
    
    # Get image dimensions
    height, width = I.shape
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Get 8-neighborhood
            p2 = I[i-1, j]
            p3 = I[i-1, j+1]
            p4 = I[i, j+1]
            p5 = I[i+1, j+1]
            p6 = I[i+1, j]
            p7 = I[i+1, j-1]
            p8 = I[i, j-1]
            p9 = I[i-1, j-1]
            
            # Convert to binary (0 or 1)
            p2 = 1 if p2 > 0 else 0
            p3 = 1 if p3 > 0 else 0
            p4 = 1 if p4 > 0 else 0
            p5 = 1 if p5 > 0 else 0
            p6 = 1 if p6 > 0 else 0
            p7 = 1 if p7 > 0 else 0
            p8 = 1 if p8 > 0 else 0
            p9 = 1 if p9 > 0 else 0
            
            # Calculate A (number of 0,1 transitions)
            A = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)
            
            # Calculate B (sum of neighbors)
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            
            # Calculate m1 and m2 based on iteration
            if iteration == 0:
                m1 = p2 * p4 * p6
                m2 = p4 * p6 * p8
            else:
                m1 = p2 * p4 * p8
                m2 = p2 * p6 * p8
            
            # Apply thinning condition
            if A == 1 and B >= 2 and B <= 6 and m1 == 0 and m2 == 0:
                M[i, j] = 1
    
    return (I & ~M)

def thinImage(originalImageArray):
    thinnedImageArray = originalImageArray.copy() / 255
    prev = numpy.zeros(originalImageArray.shape[:2], numpy.uint8)
    diff = None

    while True:
        thinnedImageArray = ZSThinningIteration(thinnedImageArray, 0)
        thinnedImageArray = ZSThinningIteration(thinnedImageArray, 1)
        diff = numpy.absolute(thinnedImageArray - prev)
        prev = thinnedImageArray.copy()
        if numpy.sum(diff) == 0:
            break

    return thinnedImageArray * 255
