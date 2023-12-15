import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
from .Mesh import Mesh
import meshio


class Edge:
    def __init__(self, n1, n2, nodes, parent, branch) -> None:
        self.n1 = n1 #ids
        self.n2 = n2 #ids

        self.dir = (nodes[n2] - nodes[n1])/np.linalg.norm(nodes[n2] - nodes[n1])
        self.parent = parent
        self.branch = branch

def interpolate(vectors, r, t):
    return t*vectors[2] + r*vectors[1] + (1-r-t)*vectors[0]

def eval_field(point, field, mesh):
    ppoint,tri,r,t = mesh.project_new_point(point, 5)
    return interpolate(field[mesh.connectivity[tri]], r, t), ppoint, tri

def point_in_mesh(point, mesh):
    point = np.append(point, np.zeros(1))
    _,tri,_,_ = mesh.project_new_point(point, 5)
    return tri >= 0

class Parameters:
    """Class to specify the parameters of the fractal tree.
            
    Attributes:
        meshfile (str): path and filename to obj file name.
        filename (str): name of the output files.
        init_node (numpy array): the first node of the tree.
        second_node (numpy array): this point is only used to calculate the initial direction of the tree and is not included in the tree. Please avoid selecting nodes that are connected to the init_node by a single edge in the mesh, because it causes numerical issues.
        init_length (float): length of the first branch.
        N_it (int): number of generations of branches.
        length (float): average lenght of the branches in the tree.
        std_length (float): standard deviation of the length. Set to zero to avoid random lengths.
        min_length (float): minimum length of the branches. To avoid randomly generated negative lengths.
        branch_angle (float): angle with respect to the direction of the previous branch and the new branch.
        w (float): repulsivity parameter.
        l_segment (float): length of the segments that compose one branch (approximately, because the lenght of the branch is random). It can be interpreted as the element length in a finite element mesh.
        Fascicles (bool): include one or more straigth branches with different lengths and angles from the initial branch. It is motivated by the fascicles of the left ventricle. 
        fascicles_angles (list): angles with respect to the initial branches of the fascicles. Include one per fascicle to include.
        fascicles_length (list): length  of the fascicles. Include one per fascicle to include. The size must match the size of fascicles_angles.
        save (bool): save text files containing the nodes, the connectivity and end nodes of the tree.
        save_paraview (bool): save a .vtu paraview file. The tvtk module must be installed.
    """
    def __init__(self):
        self.meshfile = None
        self.init_node_id= 0
        self.second_node_id = 1
        self.init_length=0.1
#Number of iterations (generations of branches)
        self.N_it=10
#Median length of the branches
        self.length=.1
#Standard deviation of the length
#Min length to avoid negative length
        self.branch_angle=0.15
        self.w=0.1
#Length of the segments (approximately, because the lenght of the branch is random)
        self.l_segment=.01

###########################################
# Fascicles data
###########################################
        self.fascicles_angles=[] #rad
        self.fascicles_length=[]

class FractalTree:
    def __init__(self, params):
        self.m = Mesh(params.meshfile)
        print('computing uv map')
        self.m.compute_uvscaling()
        self.mesh_uv = Mesh(verts = np.concatenate((self.m.uv,np.zeros((self.m.uv.shape[0],1))), axis =1), connectivity= self.m.connectivity)
        self.scaling_nodes = np.array(self.mesh_uv.tri2node_interpolation(self.m.uvscaling))
        self.params = params
    def scaling(self,x):
        x = np.append(x, np.zeros(1))
        f, _, tri = eval_field(x, self.scaling_nodes, self.mesh_uv)
        return np.sqrt(f), tri
    
    def grow_tree(self):
        branches = defaultdict(list)
        branch_id = 0

        end_nodes = []

        sister_branches = {}

        dx = self.params.l_segment
        init_node = self.mesh_uv.verts[self.params.init_node_id][:2]
        second_node = self.mesh_uv.verts[self.params.second_node_id][:2]
        init_dir = second_node - init_node
        init_dir /= np.linalg.norm(init_dir)
        s, tri = self.scaling(init_node) 
        if tri < 0:
          raise "the initial node is outside the domain"
        nodes = [init_node, init_node + s*dx*init_dir]
        edges = [Edge(0,1,nodes,None,branch_id)]


        edge_queue = [0]
        branches[branch_id].append(0)

        branch_length = self.params.length
        init_branch_length = self.params.init_length


        theta = self.params.branch_angle
        w = self.params.w

        Rplus = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        Rminus = np.array([[np.cos(-theta), -np.sin(-theta)],[np.sin(-theta), np.cos(-theta)]])




        for i in range(int(init_branch_length/dx)):

            edge_id = edge_queue.pop(0)
            edge = edges[edge_id]
            new_dir = edge.dir 
            new_dir /= np.linalg.norm(new_dir)
            s, tri = self.scaling(nodes[edge.n2]) 
            if tri < 0:
                raise "the initial branch goes out of the domain"
            new_node = nodes[edge.n2] + new_dir*dx*s
            new_node_id = len(nodes)
            nodes.append(new_node)
            branches[edge.branch].append(new_node_id)
            edge_queue.append(len(edges))
            edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, edge.branch))

        branching_edge_id = edge_queue.pop(0)


        for fascicle_length, fascicles_angle in zip(self.params.fascicles_length, self.params.fascicles_angles):
            Rotation = np.array([[np.cos(fascicles_angle), -np.sin(fascicles_angle)],[np.sin(fascicles_angle), np.cos(fascicles_angle)]])    
            edge = edges[branching_edge_id]
            new_dir = np.matmul(Rotation, edge.dir)
            new_dir /= np.linalg.norm(new_dir)
            s, tri = self.scaling(nodes[edge.n2]) 
            if tri < 0:
                raise "the fascicle goes out of the domain"
            new_node = nodes[edge.n2] + new_dir*dx*s
            new_node_id = len(nodes)
            nodes.append(new_node)
            branch_id += 1
            branches[branch_id].append(new_node_id)
            edge_queue.append(len(edges))
            edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, edge.branch))
            for i in range(int(fascicle_length/dx)):
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]
                new_dir = edge.dir 
                new_dir /= np.linalg.norm(new_dir)
                s, tri = self.scaling(nodes[edge.n2]) 
                if tri < 0:
                    raise "the fascicle goes out of the domain"
                new_node = nodes[edge.n2] + new_dir*dx*s
                new_node_id = len(nodes)
                nodes.append(new_node)
                branches[edge.branch].append(new_node_id)
                edge_queue.append(len(edges))
                edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, edge.branch))



        for gen in range(self.params.N_it):
            print('generation', gen)
            branching_queue = []
            while len(edge_queue) > 0:
                edge_id = edge_queue.pop(0)
                edge = edges[edge_id]
                for R in [Rplus, Rminus]:
                    new_dir = np.matmul(R,edge.dir)
                    new_dir /= np.linalg.norm(new_dir)
                    s, tri = self.scaling(nodes[edge.n2]) 
                    new_node = nodes[edge.n2] + new_dir*dx*s
                    if ~point_in_mesh(new_node, self.mesh_uv):
                        end_nodes.append(edge.n2)
                        continue
                    new_node_id = len(nodes)
                    nodes.append(new_node)
                    branching_queue.append(len(edges))
                    branch_id += 1
                    branches[branch_id].append(new_node_id)
                    edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, branch_id))
                sister_branches[branch_id - 1] = branch_id
                sister_branches[branch_id] = branch_id - 1


            edge_queue = branching_queue

            for i in range(int(branch_length/dx)):
                growing_queue = []

                while len(edge_queue) > 0:
                    edge_id = edge_queue.pop(0)
                    edge = edges[edge_id]
                    # collision detection
                    temp_nodes = np.array(nodes)
                    temp_nodes[branches[edge.branch]] = np.array([1e9,1e9])
                    temp_nodes[branches[sister_branches[edge.branch]]] = np.array([1e9,1e9])
                    tree = cKDTree(temp_nodes)
                    pred_node = nodes[edge.n2]# + dx*edge.dir
                    dist, closest = tree.query(pred_node)
                    s, tri = self.scaling(nodes[edge.n2]) 
                    if dist < 0.9*dx*s:
                        end_nodes.append(edge.n2)
                        continue

                    # grad calculation
                    temp_nodes = np.array(nodes)
                    temp_nodes[branches[edge.branch]] = np.array([1e9,1e9])
                    tree = cKDTree(temp_nodes)
                    pred_node = nodes[edge.n2]# + dx*edge.dir
                    dist, closest = tree.query(pred_node)
                    grad_dist = (pred_node - nodes[closest])/dist
                    new_dir = edge.dir + w*grad_dist
                    new_dir /= np.linalg.norm(new_dir)
                    new_node = nodes[edge.n2] + new_dir*dx*s
                    new_node_id = len(nodes)
                    if ~point_in_mesh(new_node, self.mesh_uv):
                        end_nodes.append(edge.n2)
                        continue
                    nodes.append(new_node)
                    growing_queue.append(len(edges))
                    branches[edge.branch].append(new_node_id)
                    edges.append(Edge(edge.n2, new_node_id,nodes, edge_id, edge.branch))
                edge_queue = growing_queue

        end_nodes += [edges[edge].n2 for edge in edge_queue]

        self.uv_nodes = np.array(nodes)
        self.edges = edges
        self.end_nodes = end_nodes

        self.connectivity = []
        for edge in edges:
            self.connectivity.append([edge.n1, edge.n2])

        self.nodes_xyz = []
        for node in nodes:

            n = np.append(node, np.zeros(1))
            f, _, tri = eval_field(n, self.m.verts, self.mesh_uv)
            self.nodes_xyz.append(f) 

    def save(self, filename):
        line = meshio.Mesh(np.array(self.nodes_xyz), [('line',np.array(self.connectivity))])
        line.write(filename)


if __name__ == '__main__':
    params = Parameters()
    params.init_node_id = 738
    params.second_node_id = 210
    params.l_segment = 0.01
    params.init_length = 0.3
    params.length= 0.15
    params.meshfile = 'data/ellipsoid.obj'
    params.fascicles_length = [20*params.l_segment, 40*params.l_segment]
    params.fascicles_angles = [-0.4, 0.5] # in radians

    tree = FractalTree(params)

    tree.grow_tree()
    tree.save('output/ellipsoid_purkinje.vtu')


