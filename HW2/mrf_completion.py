# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:35:26 2017

@author: carmonda
"""

V_MAX = 50   #this is the threshold

from scipy import misc
import numpy as np

class Vertex(object):
    def __init__(self,index=0,name='',y=None,neighs=None,in_msgs = None,observed=True,normalize_factor=None):
        self._name = name
        self._y = y # original pixel
        if(neighs == None): neighs = set() # set of neighbour nodes
        if(in_msgs==None): in_msgs = {} # dictionary mapping neighbours to their messages
        self._neighs = neighs
        self._in_msgs = in_msgs
        self._observed = observed
        self._i = index
        #self._normalize_factor = normalize_factor
    def add_neigh(self,vertex):
        self._neighs.add(vertex)
    def rem_neigh(self,vertex):
        self._neighs.remove(vertex)

    #initiaize the message sfrom self to v
    # TODO: the initialization should be done randomly or a uniform value.
    # source: http://adv-ml-2017.wikidot.com/forum/t-2228406/lbp-for-image-completion
    def init_message(self,v):
        if(self._observed):
            v._in_msgs[self] = self._y
        else:
            v._in_msgs[self] = 1

    #This is formula no.2 in the HW doc
    #source: http://adv-ml-2017.wikidot.com/forum/t-2219436/log-of-the-factors
    def calc_log_basic_msg(self,neigh,neigh_val):
        res_for_default_val = None
        res, temp = 0,0
        for i in range(256):
            if(res_for_default_val != None and neigh_val-i<50):
                temp = res_for_default_val
            else:
                temp = np.log(fi(i,neigh_val))
                for n in self._neighs:
                    #if(n==neigh):
                     #   continue
                    msg = self._in_msgs[n]
                    temp += np.log(msg)
                if(res_for_default_val==None and neigh_val-i<50):   #TODO: fix
                    res_for_default_val = temp
            if(temp>res or i==0):
                res = temp
        return res

    #recalculate the pixels after recieving the messages
    def get_belief(self):
        res = 0
        for i in range(256):
            temp = 0
            for n in self._neighs:
                temp += n._in_msgs(self) #because it's log
            if(temp > res or i==0):
                self._y = i
                res = temp


    #calculate the message and normalize it
    def get_msg(self,neigh):
        #first we calculate res
        res = self.calc_log_basic_msg(neigh,neigh._y)
        #then we normalize res
        normalize_factor = 0
        for i in range(256):
            normalize_factor += self.calc_log_basic_msg(neigh,i)
        res -= np.log(normalize_factor)
        return res

    #send message from self to neigh
    def snd_msg(self,neigh):
        """ Combines messages from all other neighbours
            to propagate a message to the neighbouring Vertex 'neigh'.
        """
        neigh._in_msgs[self] = self.get_msg(neigh)

    def __str__(self):
        ret = "Name: "+self._name
        ret += "\nNeighbours:"
        neigh_list = ""
        for n in self._neighs:
            neigh_list += " "+n._name
        ret+= neigh_list
        return ret
    
class Graph(object):
    def __init__(self, graph_dict=None,root=None,index_vertexes = None):
        """ initializes a graph object
            If no dictionary is given, an empty dict will be used
        """
        if graph_dict == None:
            graph_dict = {}
        if index_vertexes == None:
            index_vertexes = {}
        self._graph_dict = graph_dict
        self._root = root
        self._index_vertexes = index_vertexes
    def vertices(self):
        """ returns the vertices of a graph"""
        return list(self._graph_dict.keys())
    def edges(self):
        """ returns the edges of a graph """
        return self._generate_edges()
    def get_vertex(self,index):
        return self._index_vertexes[index]
    def add_vertex(self, vertex,i):
        """ If the vertex "vertex" is not in
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex]=[]
            self._index_vertexes[i] = vertex
    def add_edge(self,edge):
        """ assumes that edge is of type set, tuple, or list;
            between two vertices can be multiple edges.
        """
        edge = set(edge)
        (v1,v2) = tuple(edge)
        if v1 in self._graph_dict:
            self._graph_dict[v1].append(v2)
        else:
            self._graph_dict[v1] = [v2]
        # if using Vertex class, update data:
        if(type(v1)==Vertex and type(v2)==Vertex):
            v1.add_neigh(v2)
            v2.add_neigh(v1)
            v1.init_message(v2)
            v2.init_message(v1)

    def generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one or two vertices
        """
        e = []
        for v in self._graph_dict:
            for neigh in self._graph_dict[v]:
                if {neigh,v} not in e:
                    e.append({v,neigh})
        return e
    def __str__(self):
        res = "V: "
        for k in self._graph_dict:
            res+=str(k) + " "
        res+= "\nE: "
        for edge in self._generate_edges():
            res+= str(edge) + " "
        return res

def is_observed(row,col,x1,x2,y1,y2): #helper function for deciding which pixels are observed
    """
    Returns True/False by whether pixel at (row,col) was observed or not
    """
    #x1,x2,y1,y2 = 92,106,13,93 # unobserved rectangle borders for 'pinguin-img.png'
    if(row<x1 or row>x2): return True
    if(col<y1 or col>y2): return True
    return False

def build_grid_graph(n,m,img_mat,x1,x2,y1,y2):
    """ Builds an nxm grid graph, with vertex values corresponding to pixel intensities.
    n: num of rows
    m: num of columns
    img_mat = np.ndarray of shape (n,m) of pixel intensities
    
    returns the Graph object corresponding to the grid
    """
    V = []
    g = Graph()
    # add vertices:
    for i in range(n*m):
        row,col = (i//m,i%m)
        v = Vertex(index=i,name="v"+str(i),y=img_mat[row][col],observed = is_observed(row,col,x1,x2,y1,y2))
        g.add_vertex(v,i)
        if((i%m)!=0): # has left edge
            g.add_edge((v,V[i-1]))
        if(i>=m): # has up edge
            g.add_edge((v,V[i-m]))
        V += [v]
    g._root = g.get_vertex(0)
    return g

def grid2mat(grid,n,m):
    """ convertes grid graph to a np.ndarray
    n: num of rows
    m: num of columns
    
    returns: np.ndarray of shape (n,m)
    """
    mat = np.zeros((n,m))
    l = grid.vertices() # list of vertices
    for v in l:
        i = int(v._name[1:])
        row,col = (i//m,i%m)
        mat[row][col] = v._y # you should change this of course
    return mat

#------------------------------------------------------------
#my code:

def fi(xi,xj):
    res = np.e**(-min(abs(xi-xj),V_MAX))
    return res

#given a MRF model retuens the p(x1,...,xn) of pix_arr
def p(G,pix_arr):
    res = 1
    edges = G.edges
    for i in range(len(pix_arr)):
        for j in range(len(pix_arr)):
            if {pix_arr[i],pix_arr[j]} in edges:
                res *= fi(pix_arr[i],pix_arr[j])
    return res




#--------------------------------------------------------------

# begin:
#if len(sys.argv)<2:
#    print('Please specify output filename')
#    exit(0)
# load image:
print("Starting test...")
frame_width = 1
outfile_name = "Result"
img_path = "C:\\Users\\Yair Hadas\\Desktop\\שיטות מתקדמות בלמידה חישובית\\Adv-ML-HW\\HW2\\penguin-img.png"
image = misc.imread(img_path)
x1,x2,y1,y2 = 91,108,12,95
image_segment = image[x1:x2,y1:y2] # currently a minimum of 1 pixel frame
n,m = image_segment.shape
# build grid:
g = build_grid_graph(n,m,image_segment,0+frame_width, m-(frame_width+1), 0+frame_width, n-(frame_width+1))
print("loaded image and made a grid from it successfully")
# process grid:

for i in range(1):
    print("Started run no."+str(i+1))
    print("Sending messages...")
    for i in range(n*m):
        print("now vertex-"+str(i)+" sends it's messages")
        v = g.get_vertex(i)
        for neigh in v._neighs:
            v.snd_msg(neigh)

    print("Updating pixels values...")
    for v in g.vertices():
        if(v._observed == False):
            v.get_belief()
    print("Finisheded run no." + str(i))

# convert grid to image:
infered_img = grid2mat(g,n,m)
image_final = image
image_final[x1:x2,y1:y2] = infered_img # plug the inferred values back to the original image
# save result to output file
#outfile_name = sys.argv[1]
outfile_name  = "HW2\\Result"
misc.toimage(image_final).save(outfile_name+'.png')