from PurkinjeUV import PurkinjeTree, FractalTree, Parameters
#from FractalTree import Fractal_Tree_3D
from Endocardium_demo import EndocardialMesh
import matplotlib.pyplot as plt
import meshio
import os
import numpy as onp
import jax.numpy as np
import pickle

class BO_Purkinje():
    # Class to perform Bayesian Optimization on a Purkinje tree.
    def __init__(self, patient, meshes_list, init_length, length, w, l_segment, fascicles_length, fascicles_angles, branch_angle, N_it):
        self.patient          = patient
        self.meshes_list      = meshes_list
        self.init_length      = init_length
        self.length           = length
        self.w                = w
        self.l_segment        = l_segment
        self.fascicles_length = fascicles_length
        self.fascicles_angles = fascicles_angles
        self.branch_angle     = branch_angle
        self.N_it             = N_it

        self.LVfractaltree, self.RVfractaltree = self.initialize()



    # set initial parameters for LV and RV fractal trees
    def initialize(self):
        LV1,LV2,RV1,RV2 = self.meshes_list
        param_LV        = Parameters()
        param_LV.save   = False

        param_LV.init_length = self.init_length
        param_LV.length      = self.length
        param_LV.w           = self.w
        param_LV.l_segment   = self.l_segment

        param_LV.fascicles_length = self.fascicles_length
        param_LV.fascicles_angles = self.fascicles_angles

        param_LV.branch_angle = self.branch_angle
        param_LV.N_it         = self.N_it

        param_RV      = Parameters()
        param_RV.save = False

        param_RV.init_length = self.init_length
        param_RV.length      = self.length
        param_RV.w           = self.w
        param_RV.l_segment   = self.l_segment

        param_RV.fascicles_length = self.fascicles_length
        param_RV.fascicles_angles = self.fascicles_angles

        param_RV.branch_angle = self.branch_angle
        param_RV.N_it         = self.N_it

        # LV Purkinje
        param_LV.meshfile       = f"{self.patient}_LVendo_heart_cut.obj"
        mesh_lv                 = meshio.read(param_LV.meshfile)
        param_LV.init_node_id   = LV1
        param_LV.second_node_id = LV2
        LVfractaltree           = FractalTree(param_LV) 

        # RV Purkinje
        param_RV.meshfile       = f"{self.patient}_RVendo_heart_cut.obj"
        mesh_rv                 = meshio.read(param_RV.meshfile)
        param_RV.init_node_id   = RV1
        param_RV.second_node_id = RV2
        RVfractaltree           = FractalTree(param_RV)
        
        return LVfractaltree, RVfractaltree
    
    
    
    def run_ECG(self, LVfractaltree = None, RVfractaltree = None, n_sim = 0, modify = False, side = 'both', **kwargs):
        LVfractaltree = self.LVfractaltree if LVfractaltree is None else LVfractaltree
        RVfractaltree = self.RVfractaltree if RVfractaltree is None else RVfractaltree
        
        # if modify is True, choose which tree you want to change: LV, RV or both, and give arguments as parameter_name = value
        #if you are only performing a single simulation then n_sim=0
        if modify == True:
            for key, value in kwargs.items():
                if key != "root_time" and key != "cv":
                    if side == 'LV':
                        exec(f"LVfractaltree.params.{key} = {value}")
                    elif side == 'RV':
                        exec(f"RVfractaltree.params.{key} = {value}")
                    else:
                        for i, v in enumerate(value):
                            if i == 0:
                                exec(f"LVfractaltree.params.{key} = {v}")
                            else:
                                exec(f"RVfractaltree.params.{key} = {v}")

        LVfractaltree.grow_tree()
        LVtree = PurkinjeTree(onp.array(LVfractaltree.nodes_xyz), onp.array(LVfractaltree.connectivity), onp.array(LVfractaltree.end_nodes))

        RVfractaltree.grow_tree()
        RVtree = PurkinjeTree(onp.array(RVfractaltree.nodes_xyz), onp.array(RVfractaltree.connectivity), onp.array(RVfractaltree.end_nodes))

        ### Not using propeiko ###
        dir,model = os.path.split(self.patient)
        pat       = model.split('-')[0]
        
        Endo = EndocardialMesh(myo_mesh            = f"{dir}/{pat}_mesh_oriented.vtk",
                               electrodes_position = f"{dir}/electrode_pos.pkl",
                               fibers              = f"{dir}/{pat}_f0_oriented.vtk")

        # we set activation in {L,R} trees
        # NOTE to simulate block, set to large value
        if modify == True and "root_time" in kwargs:
            root_t              = kwargs["root_time"]
            LVroot, LVroot_time = 0, -1. * onp.min([0., root_t])
            RVroot, RVroot_time = 0, onp.max([0., root_t])
        else:
            LVroot,LVroot_time = 0, 0.0
            RVroot,RVroot_time = 0, 0.0

        # CV in the trees
        print ("Changing units ...")
        if modify == True and "cv" in kwargs:
            LVtree.cv = kwargs["cv"] # [m/s]
            RVtree.cv = kwargs["cv"] # [m/s]
        else:
            LVtree.cv = 2.5 # [m/s]
            RVtree.cv = 2.5 # [m/s]

        # we can also set PVCs in myocardium
        #PVCs,PVCs_vals = onp.array([38894]),onp.array([-100.0])
        PVCs      = onp.array([],dtype=int)
        PVCs_vals = onp.array([],dtype=float)

        # junctions (do not correspond to nodes in general)
        LVpmj      = LVtree.pmj
        RVpmj      = RVtree.pmj
        LVpmj_vals = onp.empty_like(LVpmj, dtype = float)
        RVpmj_vals = onp.empty_like(RVpmj, dtype = float)
        LVpmj_vals.fill(onp.inf)
        RVpmj_vals.fill(onp.inf)

        # initialize the coupling points
        x0      = onp.r_[ LVpmj, RVpmj, PVCs ]
        x0_xyz  = onp.r_[ LVtree.xyz[LVpmj,:], RVtree.xyz[RVpmj,:], Endo.xyz[PVCs,:] ]
        x0_side = onp.r_[ onp.repeat('L', LVpmj.size),
                        onp.repeat('R', RVpmj.size),
                        onp.repeat('M', PVCs.size) ]
        x0_vals = onp.r_[ LVpmj_vals, RVpmj_vals, PVCs_vals ]

        SIDE_LV = (x0_side == 'L')
        SIDE_RV = (x0_side == 'R')

        # smaller tolerance
        kmax    = 8
        tol_act = 0.0  # absolute error on activation at PMJs
        tol_ecg = 1e-4 # abs error on ECG

        # kmax    = 4
        # tol_act = 1.0  # absolute error on activation at PMJs
        # tol_ecg = 1e-2 # abs error on ECG

        save             = False
        ecg              = None
        # save_simulations = False # save only last iteration
        # NOTE this is always doing at least 2 iterations, unless
        # we initialize the trees before (thats cheap)

        for k in range(kmax):

            # activate Purkinje trees
            x0_vals[SIDE_LV] = LVtree.activate_fim(onp.r_[LVroot,x0[SIDE_LV]], onp.r_[LVroot_time,x0_vals[SIDE_LV]])
            x0_vals[SIDE_RV] = RVtree.activate_fim(onp.r_[RVroot,x0[SIDE_RV]], onp.r_[RVroot_time,x0_vals[SIDE_RV]])
            
            # activate the myocardium at PMJs
            myo_vals = Endo.activate_fim(x0_xyz, x0_vals, return_only_pmjs = True)

            # early sites
            nnLV = (myo_vals[SIDE_LV] - x0_vals[SIDE_LV] + tol_act < 0).sum()
            nnRV = (myo_vals[SIDE_RV] - x0_vals[SIDE_RV] + tol_act < 0).sum()
            
            print(f"Iteration {k}, nLV = {nnLV}, nRV = {nnRV}")
            
            #print(onp.c_[x0_vals, myo_vals])
            #print((myo_vals-x0_vals+tol)[SIDE_RV][myo_vals[SIDE_RV] < x0_vals[SIDE_RV]],
            #      x0[SIDE_RV][myo_vals[SIDE_RV] < x0_vals[SIDE_RV]] )

            # updates only those earlier than Purkinje
            x0_vals[SIDE_LV] = onp.minimum(x0_vals[SIDE_LV], myo_vals[SIDE_LV])
            x0_vals[SIDE_RV] = onp.minimum(x0_vals[SIDE_RV], myo_vals[SIDE_RV])

            # save if requested
            if save:
                LVtree.save(f"output/{pat}/LVtree_{k:02d}.vtu")
                LVtree.save_pmjs(f"output/{pat}/LVpmjs_{k:02d}.vtu")
                RVtree.save(f"output/{pat}/RVtree_{k:02d}.vtu")
                RVtree.save_pmjs(f"output/{pat}/RVpmjs_{k:02d}.vtu")
                Endo.save_pv(f"output/{pat}/Endo_{k:02d}.vtp")

            # NOTE: we do not check the error, because we pace the 3D model
            # at non-vertices, while the output is piecewise linear. Therefore,
            # when we read again the value after activation it differs from
            # the imposed one, because of the linear interpolation.
            # In general, interpolated value is always higher than the exact
            # solution in the voxel, so we can simply check the sign
            #
            # If some activation in myocardium is earlier than the Purkinje,
            # we need to activate once again the Purkinje

            #if nnLV + nnRV == 0:
            #    break

            # check on ECG
            ecg_new = Endo.new_get_ecg(record_array=False).copy()
        
            if ecg is not None:
                ecg_err = onp.linalg.norm(ecg-ecg_new)
                print(f"ECG error = {ecg_err}")
                if ecg_err < tol_ecg:
                    break
            
            ecg = ecg_new
        
        # if save_simulations == True: # change path to save files
        #     LVtree.save(f"/content/drive/MyDrive/Purkinje/{name_var}_{side}_results/{propeiko.fpar['prefix']}_LVtree_{n_sim:03d}.vtu")
        #     LVtree.save_pmjs(f"/content/drive/MyDrive/Purkinje/{name_var}_{side}_results/{propeiko.fpar['prefix']}_LVpmjs_{n_sim:03d}.vtu")
        #     RVtree.save(f"/content/drive/MyDrive/Purkinje/{name_var}_{side}_results/{propeiko.fpar['prefix']}_RVtree_{n_sim:03d}.vtu")
        #     RVtree.save_pmjs(f"/content/drive/MyDrive/Purkinje/{name_var}_{side}_results/{propeiko.fpar['prefix']}_RVpmjs_{n_sim:03d}.vtu")
        #     Endo.save(f"/content/drive/MyDrive/Purkinje/{name_var}_{side}_results/{propeiko.fpar['prefix']}_Endo_{n_sim:03d}.vtp")
        #     # propeiko.save(f"/content/drive/MyDrive/Purkinje/{name_var}_{side}_results/{propeiko.fpar['prefix']}_Myo_{n_sim:03d}.vtu")

        ecg = Endo.new_get_ecg()
        return ecg, Endo, LVtree, RVtree
    
    
    
    def save_fractaltrees(self, filename_LVtree, filename_RVtree):
        self.LVfractaltree.save(filename_LVtree)
        self.RVfractaltree.save(filename_RVtree)
