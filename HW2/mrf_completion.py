import sys
from scipy import misc
import numpy as np

V_MAX = 50   #this is the threshold
ITER_NO = 5
#Since we implemented the log version of the algorithm it is unnecessary to take an exponent
def fi(xi,xj):
    res = np.e**(-min(abs(xi-xj),V_MAX))
    return res

#Here we create the matrix FI which contains all the possible values of the fi function
FI = np.array([[j for i in range(256)] for j in range(256)])
FI = np.vectorize(fi)(FI,[i for i in range(256)])


class Vertex(object):
    def __init__(self,index=0,name='',y=None,neighs=None,in_msgs = None,observed=True):
        self._name = name
        self._y = y # original pixel
        if(neighs == None): neighs = set() # set of neighbour nodes
        if(in_msgs==None): in_msgs = {} # dictionary mapping neighbours to their messages
        self._neighs = neighs
        self._in_msgs = in_msgs
        self._observed = observed
        self._i = index
    def add_neigh(self,vertex):
        self._neighs.add(vertex)
    def rem_neigh(self,vertex):
        self._neighs.remove(vertex)

    #initiaize the messages from self to v
    # source: http://adv-ml-2017.wikidot.com/forum/t-2228406/lbp-for-image-completion
    def init_message(self,v):
        v._in_msgs[self] = [0 for i in range(256)]
        if(self._observed):
            v._in_msgs[self] = np.array([FI[i][self._y] for i in range(256)])
        else:
            v._in_msgs[self] = np.array([])

    #source: http://adv-ml-2017.wikidot.com/forum/t-2219436/log-of-the-factors
    def calc_log_basic_msg(self,neigh_suggested_val,neigh):
        '''
        Sends message from self to neigh.
        :return The maximum value we get from equation (2) from the homework.
        '''
        #creating matrix with all the neighbours messages
        neighs_msgs = []
        for n in self._neighs:
            if (n != neigh and len(self._in_msgs[n]) > 0):
                neighs_msgs.append(np.log(self._in_msgs[n]))
        if(len(neigh._in_msgs[self])>0):
            neighs_msgs.append(np.log(neigh._in_msgs[self]))

        #calculate for each pixel value equation 2 result
        results = np.log(FI[neigh_suggested_val]) + np.sum(neighs_msgs, axis=0)
        return np.max(results)

    #For each possible neigh value calculate the message from self to neigh
    def get_msgs(self,neigh):
        results = np.array([i for i in range(256)])
        results = np.vectorize(self.calc_log_basic_msg)(results,neigh)
        #We droped the normalization since in this implementation it's unnecessary
        return results

    #send message from self to neigh
    def snd_msg(self,neigh):
        """ Combines messages from all other neighbours
            to propagate a message to the neighbouring Vertex 'neigh'.
        """
        #in the begining the unobsereved pixels has no messages values, so we didn't want their default messages to have influence
        if not (len(np.append([],list(self._in_msgs.values()))) > 0):
            return
        results = self.get_msgs(neigh)
        res = results - np.log(sum(np.e**results))
        neigh._in_msgs[self] = np.e**res

    #recalculate the pixels after recieving the messages
    def get_belief(self):
        '''
        take all the messages from all the neighs (in this implementation the messages are vectors in size 256)
        create a matrix from them and updates self's index to that of the collumn with the largest sum.
        '''
        neighs_msgs = []
        for n in self._neighs:
            if (self._in_msgs[n] != []):
                neighs_msgs.append(self._in_msgs[n])
        results = np.sum(neighs_msgs,axis=0)    #this is a matrix at size neighs_noX256
        pix_val = np.argmax(results)
        self._y = pix_val

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
        observed = is_observed(row,col,x1,x2,y1,y2)
        v = Vertex(index=i,name="v"+str(i),y=img_mat[row][col] if observed else -1 ,observed = observed)
        g.add_vertex(v,i)
        if((i%m)!=0 and not(v._observed and V[i-1]._observed)): # has left edge
            g.add_edge((v,V[i-1]))
        if((i>=m)and not(v._observed and V[i-m]._observed)): # has up edge
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


# begin:
#if len(sys.argv)<2:
#    print('Please specify output filename')
#    exit(0)
# load image:
print("Starting test...")
frame_width = 1
turn = True
outfile_name = "Result"
img_path = "C:\\Users\\Yair Hadas\\Desktop\\שיטות מתקדמות בלמידה חישובית\\Adv-ML-HW\\HW2\\penguin-img.png"
image = misc.imread(img_path)
x1,x2,y1,y2 = 91,108,12,95
image_segment = image[x1:x2,y1:y2] # currently a minimum of 1 pixel frame
n,m = image_segment.shape
# build grid:
g = build_grid_graph(n,m,image_segment,0+frame_width, n-(frame_width+1), 0+frame_width, m-(frame_width+1))
print("loaded image and made a grid from it successfully")

# process grid:
for i in range(ITER_NO):
    print("Started run no."+str(i+1))
    print("Sending messages...")
    #We jump from pixels in the top left to the buttom right to the middle, each pixel "asks" from his neighbors to send him messages.
    #After it recieved it's messages, it's immediately recalculate his pixel value.
    for j in range(n*m):
        j = m*n-j-1 if turn else j
        v = g.get_vertex(j)
        if (not v._observed):
            print("now vertex-" + str(j) + " receives it's messages")
            for neigh in v._neighs:
                neigh.snd_msg(v)
            v.get_belief()
            print("vertex-" + str(j) + " updated it's index value to: " + str(v._y))
        turn = not turn
    '''
    print("Updating pixels values...")
    for j in range(n*m):
        v = g.get_vertex(j)
        if(v._observed == False):
            v.get_belief()
            print("vertex-" + str(j) + " updated it's index value to: " + str(v._y))
    print("Finisheded run no." + str(i))
    '''
# convert grid to image:
infered_img = grid2mat(g,n,m)
image_final = image
image_final[x1:x2,y1:y2] = infered_img # plug the inferred values back to the original image
# save result to output file
#outfile_name = sys.argv[1]
outfile_name  = "..\Result-"+str(ITER_NO)+"_with_normalization"
misc.toimage(image_final).save(outfile_name+'.png')