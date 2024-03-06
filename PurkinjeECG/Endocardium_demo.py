import vtk
import numpy as np
from vtkmodules.numpy_interface import dataset_adapter as dsa
from fimpy.solver import FIMPY
import meshio
from PurkinjeECG.PurkinjeUV.vtkutils import vtk_extract_boundary_surfaces, vtkIGBReader
import os

import pyvista as pv
import pickle
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class EndocardialMesh:
    "VTK mesh of endocardium (left, right or both)"

    SIDE_LV = 1
    SIDE_RV = 2

    def __init__(self, myo_mesh, electrodes_position, fibers):
        
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(myo_mesh)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()

        mesh_endo     = reader.GetOutput() # myocardium mesh
        self.vtk_mesh = mesh_endo

        # locator to project points onto the mesh
        loc = vtk.vtkCellLocator()
        loc.SetDataSet(self.vtk_mesh)
        loc.BuildLocator()
        self.vtk_locator = loc
        # reset activation
        dd = dsa.WrapDataObject(self.vtk_mesh)
        act = np.empty(dd.Points.shape[0])
        act.fill(np.inf)
        dd.PointData.append(act,"activation")
        self.xyz = dd.Points

        # electrodes positions
        with open(f"{electrodes_position}", "rb") as input_file:
            self.electrode_pos = pickle.load(input_file)
#             print (f"electrode_pos:{self.electrode_pos}")
        
        # fibers directions
        f0_reader = vtk.vtkDataSetReader()
        f0_reader.SetFileName(fibers)
        f0_reader.ReadAllVectorsOn()
        f0_reader.ReadAllScalarsOn()
        f0_reader.Update()
        f0 = f0_reader.GetOutput()

        dd_f0            = dsa.WrapDataObject(f0)
        xyz_f0           = dd_f0.Points

        if 'fiber' in dd_f0.PointData.keys():
            fiber_directions = dd_f0.PointData['fiber'] # dd.Points order is not the same as dd_f0.Points order
            
            # reorder fiber_directions to match order of xyz
            # Compute the distance matrix between points in 'xyz' and 'xyz_f0'
            distances = cdist(self.xyz, xyz_f0)

            # Find the index of the closest point in 'xyz_f0' for each point in 'xyz'
            closest_indices = np.argmin(distances, axis = 1)

            # # Get the closest points from 'xyz_f0'
            # closest_points = xyz_f0[closest_indices] # this should be equal to xyz

            l = fiber_directions[closest_indices]

            assert len(closest_indices) == len(set(closest_indices)), \
            "There should be no repeated elements"

            assert max(distances[np.arange(len(self.xyz)),closest_indices]) < 1e-3, \
            "The minimum distances (from a point in xyz, to the same point in xyz_f0) must be low"
            
            # transform l from PointData to CellData
            mesh           = pv.UnstructuredGrid(mesh_endo)
            mesh["l"]      = l
            mesh_cell_data = mesh.point_data_to_cell_data()
        
        elif 'fiber' in dd_f0.CellData.keys():
            # It assumes the cell order is the same in the mesh and in the fiber files
            fiber_directions = dd_f0.CellData['fiber']

            mesh_cell_data      = pv.UnstructuredGrid(mesh_endo)
            mesh_cell_data["l"] = fiber_directions

            mesh_convert_data      = pv.UnstructuredGrid(mesh_endo)
            mesh_convert_data["l"] = fiber_directions
            mesh_point_data        = mesh_convert_data.cell_data_to_point_data()
            l_vtkDataArray         = dsa.numpyTovtkDataArray(mesh_point_data["l"])
            l                      = dsa.vtkDataArrayToVTKArray(l_vtkDataArray)

        else:
            raise ValueError("Fibers directions should be named 'fiber'")
        
        # normalize data
        l_cell_norms = np.linalg.norm(mesh_cell_data['l'], axis=1, keepdims=True)
        l_cell       = mesh_cell_data['l'] / l_cell_norms
        l_cell       = l_cell.astype(np.float64) # to avoid error in FIMPY
        
        # conductivity tensor field
        sigma_il = 3.0 # mS/cm
        sigma_el = 3.0 # mS/cm
        sigma_it = 0.3 # mS/cm
        sigma_et = 1.2 # mS/cm
        
        I = np.eye(self.xyz.shape[1])
        
        alpha = 2.0  # cm ms^-1 mS^-1/2
        beta  = 800. # cm^-1 
        
        # Gi = sigma_it * I + (sigma_il - sigma_it) * l_cell[:,:,np.newaxis] @ l_cell[:,np.newaxis,:]
        # Ge = sigma_et * I + (sigma_el - sigma_et) * l_cell[:,:,np.newaxis] @ l_cell[:,np.newaxis,:]
        # self.D = alpha/np.sqrt(beta) * Gi @ np.linalg.inv(Gi+Ge) @ Ge
        
        # Without np.linalg.inv()
        sigma_mt = (sigma_et * sigma_it) / (sigma_et + sigma_it)
        sigma_ml = (sigma_el * sigma_il) / (sigma_el + sigma_il)
        Gm       = sigma_mt * I + (sigma_ml - sigma_mt) * l_cell[:,:,np.newaxis] @ l_cell[:,np.newaxis,:]
        # self.D   = alpha/np.sqrt(beta) * Gm # Units?
        self.D   = alpha**2/beta * Gm * 100. # mm^2/ms^2
        print (f"Conduction velocity in the direction of the fibers: {np.sqrt(alpha**2/beta * sigma_ml * 100.)} m/s")

        # normalize l_nodes
        l_nodes_norms = np.linalg.norm(l, axis = 1, keepdims = True)
        l_nodes       = l / l_nodes_norms
        l_nodes       = l_nodes.astype(np.float64)

        # compute Gi_nodal
        self.Gi_nodal = sigma_it * I + (sigma_il - sigma_it) * l_nodes[:,:,np.newaxis] @ l_nodes[:,np.newaxis,:] # mS cm^-1

        # Compute Gi.T * grad(Z_l) once
        self.aux_int_Vl = self.new_get_ecg_aux_Vl()
        
        # # Save fibers directions
        # dd.CellData.append(l_cell,"l_cell")
        # dd.PointData.append(l_nodes,"l_nodes")
        # self.save_pv("test_fibers.vtu")

    def find_closest_pmjs(self, pmjs):
        "Project PMJs from the tree onto the cell mesh"

        # FIXME here we could try with vtkProbeFilter.GetValidPoints
        loc = self.vtk_locator
        cellId = vtk.reference(0)
        subId  = vtk.reference(0)
        d = vtk.reference(0.0)
        ppmjs = np.zeros_like(pmjs)
        for k in range(pmjs.shape[0]):
            loc.FindClosestPoint(pmjs[k,:], ppmjs[k,:], cellId, subId, d)

        return ppmjs

    def probe_activation(self, x0):
        "Find the activation at selected locations"

        x0 = self.find_closest_pmjs(x0)

        # vtk data for the junctions
        vtk_points = vtk.vtkPoints()
        for p in x0:
            vtk_points.InsertNextPoint(p)
        vtk_poly = vtk.vtkPolyData()
        vtk_poly.SetPoints(vtk_points)

        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(self.vtk_mesh)
        probe.SetInputData(vtk_poly)
        probe.Update()

        pout = dsa.WrapDataObject(probe.GetOutput())
        act = pout.PointData['activation']

        return act

    def activate_fim(self, x0, x0_vals, return_only_pmjs = False):
        "Compute activation only in the endocardium with fim-python"
            
        if return_only_pmjs:
            x0_purkinje = x0.copy()

        # cells and points of the mesh
        dd = dsa.WrapDataObject(self.vtk_mesh) # PolyData object
        # cells = dd.Polygons.reshape((-1,4))[:,1:] # triangles
        cells = dd.Cells.reshape((-1,5))[:,1:] # tetra, dd.Cells includes the type of element
        xyz   = dd.Points     

#         # conduction velocity in myocardium (isotropic)
#         ve = np.ones(cells.shape[0])
#         D  = self.cv * np.eye(xyz.shape[1])[np.newaxis] * ve[..., np.newaxis, np.newaxis]

        # initial activation
        act = np.empty(xyz.shape[0])
        act.fill(np.inf)

        # here we look for the closest cell in the mesh
        # then we activate all the neighbors with exact
        # solution
        cellId = vtk.reference(0)
        subId  = vtk.reference(0)
        x_proj = [0.0,0.0,0.0]
        dist   = vtk.reference(0.0)

        for k in range(x0.shape[0]):
            x_orig   = x0[k,:]
            self.vtk_locator.FindClosestPoint(x_orig, x_proj, cellId, subId, dist)
            Gcell    = np.linalg.inv(self.D[cellId,...])
            cell_pts = cells[cellId,:]
            # NOTE the [k] instead of k is important, otherwise broadcast is wrong!
            # that is only because v is (3,3) and x0 is (3,), while it should be (1,3)
            # in the case of v (n,3) and x0 (3,) it behaves correctly
            v             = xyz[cell_pts,:] - x0[[k],:]
            new_act       = x0_vals[k] + np.sqrt( np.einsum('ij,kj,ki->k',Gcell,v,v) )
            act[cell_pts] = np.minimum(new_act, act[cell_pts])

        # solve in the rest of the tissue
        x0      = np.isfinite(act) # fimpy can receive it as int (index ids) or boolean
        x0_vals = act[x0]
        fim     = FIMPY.create_fim_solver(xyz, cells, self.D)
        act     = fim.comp_fim(x0, x0_vals) # activation in xyz

        # update activation in VTK
        dd.PointData['activation'][:] = act
        
        if return_only_pmjs:
            # Return activation (from FIMPY) on x0 (pmjs, points from the Tree)
            x0_pv  = pv.PolyData(x0_purkinje)
            result = x0_pv.sample(pv.UnstructuredGrid(self.vtk_mesh))
            return result['activation']
        else:
            # Return activation om myocardium
            return act

    def save_meshio(self,fname,point_data=None,cell_data=None):
        "Save with meshio"

        dd = dsa.WrapDataObject(self.vtk_mesh)
        cells = dd.Polygons.reshape((-1,4))[:,1:]
        xyz   = dd.Points

        mesh  = meshio.Mesh(points=xyz, cells={'triangle':cells})
        mesh.point_data = point_data or {}
        mesh.cell_data  = cell_data or {}

        mesh.write(fname)

    def save(self,fname):
        "Save endocardium with activation"        
        
        writer = vtk.vtkXMLUnstructuredGridWriter()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(self.vtk_mesh)
        writer.SetFileName(fname)
        writer.Update()
    
    def save_pv(self,fname):
        "Save with pyvista"

        save_mesh = pv.UnstructuredGrid(self.vtk_mesh)
        save_mesh.save(fname)
        
    def new_get_ecg(self, record_array = True):
        "Obtain ecg from activation times"
        
        # read activation
        dd    = dsa.WrapDataObject(self.vtk_mesh)
        u     = dd.PointData['activation'] # n_nodes values
        cells = dd.Cells.reshape((-1,5))[:,1:] # tetra, dd.Cells includes the type of element
        
        # action potential
        V0, V1 = -80, 20 # mV
        eps    = 1.0 # ms

        umin, umax = np.min(u), np.max(u)
        uu         = u - umin
                
        req_time_ini = -5.
        req_time_fin = 200. # 5. + np.ceil(umax - umin)
        n_times      = int(req_time_fin - req_time_ini + 1)

        V_l_dict = {key: np.empty(n_times) for key in self.electrode_pos.keys()}
        
        save_Vm    = False
        save_gradU = False

        for n_t, req_time in enumerate(np.linspace(req_time_ini, req_time_fin, n_times)):
            Vm = V0 + (V1-V0)/2*(1+np.tanh( (req_time - uu)/eps ))
            
            if save_Vm:
                dd.PointData.append(Vm, f"Vm_{req_time:03d}")

            mesh      = pv.UnstructuredGrid(self.vtk_mesh)
            mesh["U"] = Vm
            
            # working with nodal values
            mesh_grad = mesh.compute_derivative(scalars='U')
            grad_U    = mesh_grad["gradient"]
            
            mesh_aux  = pv.UnstructuredGrid(self.vtk_mesh)
            mesh_aux.point_data.clear()
            mesh_aux.cell_data.clear()
            
            for electrode_name in self.electrode_pos.keys():
                V_l_nodes = np.einsum('ij,ij->i', grad_U, self.aux_int_Vl[electrode_name])
                mesh_aux[electrode_name] = V_l_nodes

            # compute integral
            V_l_integrate = mesh_aux.integrate_data()
            for electrode_name in self.electrode_pos.keys():
                V_l_dict[electrode_name][n_t] = V_l_integrate[electrode_name][0]
            
            if save_gradU:
                dd.PointData.append(grad_U, f"U_grad_{n_t:03d}")
 
        if save_Vm or save_gradU:
            self.save_pv("test_vm.vtu")

        leads_names   = ["E1",  "E2",  "E3",
                         "aVR", "aVL", "aVF",
                         "V1",  "V2",  "V3", "V4", "V5", "V6"] # list(ecg_pat.keys())
#         formats = ['f8'] * len(names)
#         dtype = dict(names = leads_names) #, formats=formats)

        V_W = 1./3. * (V_l_dict["RA"] + V_l_dict["LA"] + V_l_dict["LL"])
        
        arrays = []
        for l_name in leads_names:
            if l_name == "E1":
                arrays.append(V_l_dict["LA"] - V_l_dict["RA"])
            elif l_name == "E2":
                arrays.append(V_l_dict["LL"] - V_l_dict["RA"])
            elif l_name == "E3":
                arrays.append(V_l_dict["LL"] - V_l_dict["LA"])
            elif l_name == "aVR":
                arrays.append(3./2. * (V_l_dict["RA"] - V_W))
            elif l_name == "aVL":
                arrays.append(3./2. * (V_l_dict["LA"] - V_W))
            elif l_name == "aVF":
                arrays.append(3./2. * (V_l_dict["LL"] - V_W))
            else:
                arrays.append(V_l_dict[l_name] - V_W)
        
        if record_array:
            ecg_pat_array = np.rec.fromarrays(arrays, names = leads_names)
        else:
            ecg_pat_array = np.asarray(arrays)
            
        ### plot ecg ###
        # ecg_pat_array_plot = np.rec.fromarrays(arrays, names = leads_names)
        # fig,axs = plt.subplots(3,4,figsize=(10,13),dpi=120,sharex=True,sharey=True)
        # for ax,l in zip(axs.ravel(),ecg_pat_array_plot.dtype.names):
        #     ax.plot(np.linspace(req_time_ini, req_time_fin, n_times), ecg_pat_array_plot[l],'b', alpha=0.6) #, label="Ground truth")
        #     ax.grid(linestyle='--',alpha=0.4)
        #     ax.set_title(l)
        #     if l == "V2":
        #         ax.legend(fontsize="8")
        # fig.tight_layout()
        # plt.show()
        ### plot ecg ###
    
        return ecg_pat_array
    
    def new_get_ecg_aux_Vl(self):
        "Find Gi.T * grad(Z_l), that can be computed once"

        dd  = dsa.WrapDataObject(self.vtk_mesh)
        xyz = dd.Points
        
        aux_int_l = {}
        for electrode_name, electrode_coords in self.electrode_pos.items():
            r       = xyz - np.array(electrode_coords)
            r_norm3 = r / np.linalg.norm(r, axis = 1)**3 # ok    

            Gi_nodal_T                 = np.transpose(self.Gi_nodal, axes=(0,2,1))            
            aux_int_l[electrode_name]  = np.sum(Gi_nodal_T * r_norm3, axis = 1) # ok
        
        return aux_int_l
