"""Simple implementation of an IGB file reader.
"""
import numpy as np

def read_igb_header(filename):
    "Read and parse the header of the IGB file"

    if filename is None:
        raise RuntimeError("No filename specified")

    # the header is the first 1024-chunk of the file
    # it's just a string, easy to read and parse
    #
    with open(filename,"rb") as pdata:
        buf = pdata.read(1024)

        hdr = buf.decode().split('\r\n')
        cmm = [l.strip()[2:] for l in hdr if l.startswith("#")]
        hdr = sum((l.split() for l in (l.strip() for l in hdr if not l.startswith("#")) if len(l)>0),[])
        hdr = dict(tuple(ob.split(":")) for ob in hdr)
        # FIXME endianness
        hdr['comments'] = cmm
        for a in 'xyzt':
            if a in hdr: hdr[a] = int(hdr[a])
        for a in ['zero','facteur']:
            if a in hdr: hdr[a] = float(hdr[a])

    return hdr


def read_igb(filename,convert_to_float=False,return_header=False):
    "Read IGB file into numpy format"

    dtypes = { 'byte':  np.uint8,   'char': np.int8,
               'short': np.int16,   'long': np.int32,
               'float': np.float32, 'double': np.float64 }

    hdr = read_igb_header(filename)
    nx,ny,nz = hdr['x'],hdr['y'],hdr['z']
    nt = hdr.get('t',1)
    shape = (nt,nz,ny,nx) if nt > 1 else (nz,ny,nx)
    dtype = dtypes[hdr['type']]
    
    data = np.fromfile(filename,dtype=dtype,count=nx*ny*nz*nt,offset=1024).reshape(shape)

    if convert_to_float:
        data = hdr.get('facteur',1.0)*data + hdr.get('zero',0.0)

    if return_header:
        return data, hdr
    else:   
        return data
