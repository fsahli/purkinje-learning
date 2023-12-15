"""This file provides the implementation of a Purkinje tree using an eikonal solver
"""
from collections import Counter
from itertools import chain
from fimpy.solver import FIMPY
import meshio
from vtkutils import *
import numpy as np

class PurkinjeTree:
    "Class for eikonal solver on Purkinje tree"

    def __init__(self, nodes, connectivity, end_nodes):
        "Init from FractalTree generator"

        self.connectivity = connectivity
        self.xyz   = np.array(nodes)

        # We keep the tree in VTK for data transfer

        self.vtk_tree = vtk_unstructuredgrid_from_list(self.xyz,self.connectivity,vtk.VTK_LINE)


        # reset activation
        act = np.empty(len(self.xyz))
        act.fill(np.inf)
        d = dsa.WrapDataObject(self.vtk_tree)
        d.PointData.append(act,"activation")

        # save PMJs
        self.pmj = end_nodes

        # conduction velocity
        self.cv = 2.5  # [m/s]

    def activate_fim(self, x0, x0_vals, return_only_pmj=True):
        "Activate tree with fim-python"

        xyz = self.xyz
        elm = self.connectivity

        ve = np.ones(elm.shape[0])
        D = self.cv * np.eye(xyz.shape[1])[np.newaxis] * ve[..., np.newaxis, np.newaxis]

        fim = FIMPY.create_fim_solver(xyz,elm,D,device='cpu')
        act = fim.comp_fim(x0, x0_vals)

        # update activation in VTK
        da = dsa.WrapDataObject(self.vtk_tree)
        da.PointData['activation'][:] = act

        if return_only_pmj:
            return act[self.pmj]
        else:
            return act

    # def activate(self, x0, x0_vals, tol=1e-8, return_only_pmj=True):
    #     "Compute activation in the tree"

    #     xyz = self.xyz

    #     # initialize the data structure
    #     act = np.empty(xyz.shape[0])
    #     act.fill(np.inf)

    #     # set initial conditions
    #     act[x0] = x0_vals

    #     # we assume no activation in the middle of the branch
    #     while True:
    #         act_old = act.copy()
    #         # iterate over the branches
    #         # NOTE this could be done in parallel
    #         for branch in self.branches.values():
    #             # points in the branch
    #             bn = branch.nodes
    #             bp = xyz[bn,:]
    #             # length of each segment
    #             le = np.linalg.norm(np.diff(bp,axis=0),axis=1)
    #             # activation from first node
    #             dl = np.r_[0.0,np.cumsum(le)] / self.cv
    #             # update all nodes in the branch from left and right
    #             act[bn] = np.minimum( dl + act[bn[0]], dl[::-1] + act[bn[-1]] )

    #         err = np.linalg.norm(act-act_old)
    #         if err < tol:
    #             break

    #     # update activation in VTK
    #     da = dsa.WrapDataObject(self.vtk_tree)
    #     da.PointData['activation'][:] = act

    #     if return_only_pmj:
    #         return act[self.pmj]
    #     else:
    #         return act

    def save(self,fname):
        "Save to VTK"

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self.vtk_tree)
        writer.Update()

    def save_pmjs(self,fname):
        "Save the junctions as VTP"

        xyz = self.xyz[self.pmj]
        da = dsa.WrapDataObject(self.vtk_tree)
        act = da.PointData['activation']

        mesh  = meshio.Mesh(points=xyz, cells={'vertex':np.arange(xyz.shape[0])[:,np.newaxis]})
        mesh.point_data = {"activation": act[self.pmj]}
        #mesh.cell_data  = cell_data or {}

        mesh.write(fname)

    def get_pmjs_activation(self):
        "Return the current activation values at PMJs"

        da = dsa.WrapDataObject(self.vtk_tree)
        act = da.PointData['activation']
        return act[self.pmj]


    def save_meshio(self,fname,point_data=None,cell_data=None):
        "Save with meshio"

        xyz = self.xyz
        edges = self.extract_edges()
        mesh  = meshio.Mesh(points=xyz, cells={'line':edges})
        mesh.point_data = point_data or {}
        mesh.cell_data  = cell_data or {}

        mesh.write(fname)

    def extract_edges(self):
        "List of edges from branches"

        # edges in each branch
        bedges = chain.from_iterable(zip(b.nodes[0:-1],b.nodes[1:]) for b in self.branches.values() if len(b.nodes) > 1)
        # collect all edges
        edges = np.array(list(bedges))

        return edges

    def extract_pmj_counter(self):
        "Pure Python version"

        t = chain.from_iterable( (b.nodes[0],b.nodes[-1]) for b in self.branches.values() if len(b.nodes) > 1 )
        c = Counter(t)
        enodes = [k for k,v in c.items() if v == 1 and v != self.branches[0].nodes[0]]

        return enodes

    def extract_pmj_np_bincount(self):
        "End-nodes of the tree or junctions"

        t = chain.from_iterable( (b.nodes[0],b.nodes[-1]) for b in self.branches.values() if len(b.nodes) > 1 )
        c = np.bincount(np.fromiter(t,dtype=int))
        enodes = np.where(c == 1)[0]
        # we remove the entry point
        enodes = np.delete(enodes, self.branches[0].nodes[0])

        return enodes

    def extract_pmj_np_unique(self):

        t = chain.from_iterable( (b.nodes[0],b.nodes[-1]) for b in self.branches.values() if len(b.nodes) > 1 )
        nn,cnt = np.unique(np.fromiter(t,dtype=int), return_counts=True)
        enodes = nn[cnt==1]
        enodes = np.delete(enodes, self.branches[0].nodes[0])

        return enodes

if __name__ == "__main__":

    from FractalTreeUV import FractalTree, Parameters 
    params = Parameters()
    params.init_node_id = 738
    params.second_node_id = 210
    params.l_segment = 0.01
    params.init_length = 0.3
    params.length= 0.15
    params.meshfile = '../data/ellipsoid.obj'
    params.fascicles_length = [20*params.l_segment, 40*params.l_segment]
    params.fascicles_angles = [-0.4, 0.5] # in radians

    tree = FractalTree(params)

    tree.grow_tree()


    Ptree = PurkinjeTree(np.array(tree.nodes_xyz), np.array(tree.connectivity), np.array(tree.end_nodes))
    act = Ptree.activate_fim([0],[0.0], return_only_pmj=False)
    pmj = Ptree.pmj
    print( act[pmj] )

    Ptree.save("../output/ellipsiod_purkinje_AT.vtu")
