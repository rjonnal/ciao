# The next approach attempts to vectorize as much as possible by
# pre-generating a set of masks, one for each search box. The memory
# footprint of this approach will be much bigger, but if the masks are
# 8-bit (smallest addressable data type, I think), maybe it won't be too
# horrible. Again, do all the initialization outside of the main function so
# it doesn't count against the overhead.

def get_search_box_masks():
    masks = []
    for x,y in zip(x_ref,y_ref):
        mask = np.zeros(spots_image.shape,dtype=np.uint8)
        xr,yr = int(round(x)),int(round(y))
        y1 = yr-search_box_size//2
        y2 = yr+search_box_size//2+1
        x1 = xr-search_box_size//2
        x2 = xr+search_box_size//2+1
        mask[y1:y2,x1:x2] = 1
        masks.append(mask)
    sy,sx = spots_image.shape
    XX,YY = np.meshgrid(range(sx),range(sy))
    plt.imshow(XX)
    plt.show()
    plt.imshow(YY)
    plt.show()
    
    return masks,XX,YY

def masks_approach(masks,XX,YY):
    x_prod = XX*spots_image
    y_prod = YY*spots_image
    for index,(x,y,mask) in enumerate(zip(x_ref,y_ref,masks)):
        x_numerator = np.sum(mask*x_prod)
        y_numerator = np.sum(mask*y_prod)
        denominator = np.sum(mask*spots_image)
        x_output[index] = x_numerator/denominator - x
        y_output[index] = y_numerator/denominator - y
    
        

def time_func(f,reps=3,args=()):
    times = []
    for k in range(reps):
        t0 = time.time()
        f(*args)
        times.append(time.time()-t0)
        
    times_str = ', '.join(['%0.3f s'%t for t in times])
    report = '%s times: %s (average %0.3f s)'%(f.__name__,times_str,np.mean(times))
    print report
    
