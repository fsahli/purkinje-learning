"""Interface to some VTK functions.
"""
import vtk
import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray,numpy_to_vtk
from igbutils import read_igb, read_igb_header

def vtk_unstructuredgrid_from_list(xyz,cells,vtk_type):
    "Creates VTK ugrid from list of points and edges"

    # points directly from array
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(xyz))
        
    # VTK cells (assuming ncells x nverts, no mixed cells)
    conn = np.empty(shape=(cells.shape[0],cells.shape[1]+1), dtype=int)
    conn[:,0]  = cells.shape[1] # number of points
    conn[:,1:] = cells
    vtk_conn = numpy_to_vtkIdTypeArray(conn.ravel())
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(cells.shape[0],vtk_conn)
    
    # VTK grid
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    grid.SetCells(vtk_type,vtk_cells)

    return grid


def vtkIGBReader(fname,name="cell",cell_centered=True,scale=1.0,origin=0.0):
    "Convert to VTK image"

    # first we check the header
    hdr = read_igb_header(fname)

    # set up the reader
    igb_reader = vtk.vtkImageReader()
    igb_reader.SetFileName(fname)
    igb_reader.SetHeaderSize(1024)
    igb_reader.SetFileDimensionality(3)
    igb_reader.SetDataExtent(0,hdr['x']-1,0,hdr['y']-1,0,hdr['z']-1)
    igb_reader.SetDataSpacing(scale,scale,scale)
    igb_reader.SetDataOrigin(origin,origin,origin)
    if hdr['type'] == 'float':
        igb_reader.SetDataScalarTypeToFloat()
    elif hdr['type'] == 'byte':
        igb_reader.SetDataScalarTypeToUnsignedChar()
    elif hdr['type'] == 'short':
        igb_reader.SetDataScalarTypeToShort()
    else:
        raise NotImplementedError

    igb_reader.FileLowerLeftOn()
    igb_reader.SetScalarArrayName(name)
    igb_reader.UpdateWholeExtent()

    # vtkImageReader assumes that data is point-centered, which is
    # not the case for cell-like arrays (for instance, cell or angles).
    # There is no VTK filter I'm aware of that does the transformation
    # (note that vtkPointDataToCellData is not what we need)
    # Therefore, we do it manually
    if cell_centered:
        nx,ny,nz = hdr['x'],hdr['y'],hdr['z']
        vtk_img = vtk.vtkImageData()
        vtk_img.SetOrigin(origin,origin,origin)
        vtk_img.SetSpacing(scale,scale,scale)
        vtk_img.SetExtent(0,nx,0,ny,0,nz)
        cdata = vtk_img.GetCellData()
        pdata = igb_reader.GetOutput().GetPointData()
        numCells = vtk_img.GetNumberOfCells()
        cdata.InterpolateAllocate(pdata,numCells,1000,False)
        ext = igb_reader.GetOutput().GetExtent()
        cdata.CopyStructuredData(pdata,ext,ext)
    else:
        vtk_img = igb_reader.GetOutput()

    return vtk_img


def vtk_extract_boundary_surfaces(vtk_cell, triangulate=False):
    "Extract left/right endocardium from the cell file"

    # use Extent and not Dimensions because data are cell-centered
    nx,ny,nz = vtk_cell.GetExtent()[1::2]

    BLOOD_CODE = 104

    # extract the blood pool (left and right)
    # NOTE this is slow in general, because generates a mesh of
    # all integral voxels of the blood pool
    thres = vtk.vtkThreshold()
    thres.SetInputData(vtk_cell)
    thres.SetLowerThreshold(BLOOD_CODE)
    thres.SetUpperThreshold(BLOOD_CODE)
    thres.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,"cell")

    # extract the boundary of the blood
    geom = vtk.vtkGeometryFilter()
    geom.SetInputConnection(thres.GetOutputPort())
    geom.Update()

    #cell = np.swapaxes(dsa.WrapDataObject(vtk_cell).CellData['cell'].reshape((nz,ny,nx)),0,2)
    cell = dsa.WrapDataObject(vtk_cell).CellData['cell'].reshape((nz,ny,nx))

    #return cell

    slv = [1,99,101,121]
    srv = [100,111,112,122]

    # now we color the points depending on the side
    surf = dsa.WrapDataObject(geom.GetOutput())
    pts  = np.rint(surf.Points).astype(int)

    neig = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]]

    side = np.zeros((pts.shape[0],2),dtype=bool)
    for nn in neig:
        # cubes in one direction
        cubs = pts - nn
        # exclude boundary cubes
        val  = np.logical_and(cubs<[nx,ny,nz], cubs>=[0,0,0]).all(axis=1)
        cubs = cubs[val,:]
        # get voxel idx
        #v = cell[cubs[:,0],cubs[:,1],cubs[:,2]]
        v = cell[cubs[:,2],cubs[:,1],cubs[:,0]]
        # check if is cube (therefore inside the domain)
        side[val,0] |= np.isin(v,slv)
        side[val,1] |= np.isin(v,srv)

    sside = np.zeros(pts.shape[0],dtype=int)
    sside[side[:,0]] = 1  # lv
    sside[side[:,1]] = 2  # rv
    surf.PointData.append(sside,"side")

    # keep only LV and RV endocardia
    thresv = vtk.vtkThreshold()
    thresv.SetInputData(surf.VTKObject)
    thresv.SetLowerThreshold(1)
    thresv.SetUpperThreshold(2)
    thresv.SetInputArrayToProcess(0,0,0,vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,"side")

    geomv = vtk.vtkGeometryFilter()
    geomv.SetInputConnection(thresv.GetOutputPort())
    geomv.Update()

    # triangulation useful for solving with fim-python
    if triangulate:
        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(geomv.GetOutputPort())
        return tri.GetOutput()
    else:
        return geomv.GetOutput()

if __name__ == "__main__":

    fname_cell = "data/crt012-01-heart1mm-e.igb"
    vtk_cell = vtkIGBReader(fname_cell, cell_centered=True)
    print(vtk_cell.GetDimensions())

    vtk_extract_boundary_surfaces(vtk_cell)
