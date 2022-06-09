import numpy as np
import blosc
import pickle

def compress(file, arr,clevel=3,cname='lz4',shuffle=1):
    """
    file           path to file
    arr            numpy nd-array
    clevel         0..9
    cname          blosclz,lz4,lz4hc,snappy,zlib
    shuffle        0-> no shuffle, 1->shuffle,2->bitshuffle
    """
    max_blk_size=100_000_000 #100 MB 

    arr = np.ascontiguousarray(arr)
    shape=arr.shape
    #dtype np.object is not implemented
    if arr.dtype==np.object:
        raise(TypeError("dtype np.object is not implemented"))

    #Handling of fortran ordered arrays (avoid copy)
    is_f_contiguous=False
    if arr.flags['F_CONTIGUOUS']==True:
        is_f_contiguous=True
        arr=arr.T.reshape(-1)
    else:
        arr=np.ascontiguousarray(arr.reshape(-1))

    #Writing
    max_num=max_blk_size//arr.dtype.itemsize
    num_chunks=arr.size//max_num

    if arr.size%max_num!=0:
        num_chunks+=1

    f=open(file,"wb")
    pickle.dump((shape,arr.size,arr.dtype,is_f_contiguous,num_chunks,max_num),f)
    size=np.empty(1,np.uint32)
    num_write=max_num
    for i in range(num_chunks):
        if max_num*(i+1)>arr.size:
            num_write=arr.size-max_num*i
        c = blosc.compress_ptr(arr[max_num*i:].__array_interface__['data'][0], num_write, 
                               arr.dtype.itemsize, clevel=clevel,cname=cname,shuffle=shuffle)
        size[0]=len(c)
        size.tofile(f)
        f.write(c)
    f.close()

def decompress(file,prealloc_arr=None):
    f=open(file,"rb")
    shape,arr_size,dtype,is_f_contiguous,num_chunks,max_num=pickle.load(f)

    if prealloc_arr is None:
        arr=np.empty(arr_size,dtype)
    else:
        arr=np.frombuffer(prealloc_arr.data, dtype=dtype, count=arr_size)

    for i in range(num_chunks):
        size=np.fromfile(f,np.uint32,count=1)
        c=f.read(size[0])
        blosc.decompress_ptr(c, arr[max_num*i:].__array_interface__['data'][0])
    f.close()

    #reshape
    if is_f_contiguous:
        arr=arr.reshape(shape[::-1]).T
    else:
        arr=arr.reshape(shape)
    return arr