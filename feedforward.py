import math
from genome import DefaultGenome

best_genome = None

class FeedForwardNetwork(object):
    def __init__(self,inputs,outputs,node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key,0.0) for key in inputs+outputs)
        
    @staticmethod
    def eval_genomes(genomes,num_inputs,num_outputs,xor_inputs,xor_outputs):
        for genome_id,genome in genomes:
            genome.fitness = 4.0
            net = FeedForwardNetwork.create(genome,num_inputs,num_outputs)
            for xi,xo in zip(xor_inputs,xor_outputs):
                output = net.activate(xi)
                genome.fitness -= (output[0]-xo[0])**2
                
    def activate(self,inputs):
        for k,v in zip(self.input_nodes,inputs):
            self.values[k] = v
            
        for node,act_func,agg_func,bias,response,links in self.node_evals:
            node_inputs = []
            for i,w in links:
                node_inputs.append(self.values[i]*w)
            s = sum(node_inputs)
            z = max(-60.0,min(60.0,5.0*(bias+response*s)))
            self.values[node] = 1.0/(1+math.exp(-z))
            
        return [self.values[i] for i in self.output_nodes]
    
    @staticmethod
    def feed_forward_layers(inputs,outputs,connections):
        required = FeedForwardNetwork.required_for_output(inputs,outputs,connections)
        layers = []
        s = set(inputs)
        while 1:
            c = set(b for (a,b) in connections if a in s and b not in s)
            t = set()
            for n in c:
                if n in required and all(a in s for(a,b) in connections if b == n):
                    t.add(n)
            if not t:
                break
            layers.append(t)
            s = s.union(t)
            
        return layers
    
    @staticmethod
    def required_for_output(inputs,outputs,connections):
        required = set(outputs)
        s = set(outputs)
        while 1:
            t = set(a for (a,b) in connections if b in s and a not in s)
            if not t:
                break
            layer_nodes = set(x for x in t if x not in inputs)
            if not layer_nodes:
                break
            required = required.union(layer_nodes)
            s = s.union(t)
            
        return required
    
    @staticmethod
    def create(genome,num_inputs,num_outputs):
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]
        output_keys = [i for i in range(num_outputs)]
        input_keys = [-i-1 for i in range(num_inputs)]
        layers = FeedForwardNetwork.feed_forward_layers(input_keys,output_keys,connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode,onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode,cg.weight))
                        
                ng = genome.nodes[node]
                node_evals.append((node,'sigmoid','sum',ng.bias,1.0,inputs))
                
        return FeedForwardNetwork(input_keys,output_keys,node_evals)