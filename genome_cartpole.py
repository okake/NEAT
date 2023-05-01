from itertools import count
from random import random,gauss

class DefaultGenome(object):
    def __init__(self,key):
        self.num_inputs = 4
        self.num_outputs = 2
        self.key = key
        self.connections = {}
        self.nodes = {}
        self.fitness = None
        self.node_indexer = None
        
    def create_new(genome_type,num_genomes,num_inputs,num_outputs):
        new_genomes = {}
        genome_indexer = count(1)
        output_keys = [i for i in range(num_outputs)]
        for i in range(num_genomes):
            key = next(genome_indexer)
            g = genome_type(key)
            for node_key in output_keys:
                g.nodes[node_key] = DefaultGenome.create_node(node_key)
            for input_id,output_id in DefaultGenome.compute_full_connections(g,num_inputs,num_outputs):
                connection = DefaultGenome.create_connection(input_id,output_id)
                g.connections[connection.key] = connection
            new_genomes[key] = g
        return new_genomes
    
    def create_node(node_id):
        node = DefaultNodeGene(node_id)
        node.init_attributes()
        return node
    
    def compute_full_connections(g,num_inputs,num_outputs):
        output_keys = [i for i in range(num_outputs)]
        input_keys = [-i-1 for i in range(num_inputs)]
        connections = []
        for input_id in input_keys:
            for output_id in output_keys:
                connections.append((input_id,output_id))
                
        return connections
    
    def create_connection(input_id,output_id):
        connection = DefaultConnectionGene((input_id,output_id))
        connection.init_attributes()
        return connection
    
    def size(self):
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes),num_enabled_connections
    
    def __str__(self):
        s = "key: {0}\nFitness: {1}\nNodes:".format(self.key,self.fitness)
        for k,ng in self.nodes.items():
            s += "\n\t{0} {1!s}".format(k,ng)
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t"+str(c)
        return s
    
class BaseGene(object):
    def __init__(self,key):
        self.compatibility_weight_coefficient = 0.5
        self.key = key
        
    def __str__(self):
        attrib = ['key']+[a for a in self._gene_attributes]
        attrib = ['{0}={1}'.format(a,getattr(self,a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__,",".join(attrib))
    
    def __lt__(self,other):
        return self.key < other.key
    
    def mutate(self):
        for a in self._gene_attributes:
            v = getattr(self,a)
            if a == 'enabled':
                r = random()
                if r < 0.01:
                    v = random() < 0.5
                setattr(self,a,v)
            else:
                r = random()
                if r < 0.75:
                    v = v+gauss(0.0,0.5)
                else:
                    if r < 0.1+0.75:
                        v = gauss(0,1)
                setattr(self,a,v)
                
    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene,a,getattr(self,a))
            
        return new_gene
    
    def crossover(self,gene2):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() > 0.5:
                setattr(new_gene,a,getattr(self,a))
            else:
                setattr(new_gene,a,getattr(gene2,a))
                
        return new_gene
    
class DefaultNodeGene(BaseGene):
    _gene_attributes = ['bias']
    
    def __init__(self,key):
        BaseGene.__init__(self,key)
        
    def init_attributes(self):
        self.bias = gauss(0,1)
    
    def distance(self,other):
        d = abs(self.bias-other.bias)
        return d*self.compatibility_weight_coefficient
    
class DefaultConnectionGene(BaseGene):
    _gene_attributes = ['weight','enabled']
    
    def __init__(self,key):
        BaseGene.__init__(self,key)
        
    def init_attributes(self):
        self.weight = gauss(0,1)
        self.enabled = True
        
    def distance(self,other):
        d = abs(self.weight-other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d*self.compatibility_weight_coefficient