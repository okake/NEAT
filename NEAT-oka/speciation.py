from itertools import count
from random import random,gauss
from genome import DefaultGenome

compatibility_threshold = 3.0
initial_connection = 'full'
compatibility_disjoint_coefficient = 1.0

class Species(object):
    def __init__(self,key,generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None
        self.members = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []
        
    def update(self,representative,members):
        self.representative = representative
        self.members = members
        
    def get_fitnesses(self):
        return [m.fitness for m in self.members.values()]
    

class DefaultSpeciesSet(object):
    def __init__(self):
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}
        
    @staticmethod
    def distance(me,other):
        node_distance = 0.0
        if me.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in me.nodes:
                    disjoint_nodes += 1
            
            for k1,n1 in me.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    distance_nodes += 1
                else:
                    node_distance += n1.distance(n2)
                    
            max_nodes = max(len(me.nodes),len(other.nodes))
            node_distance = (node_distance+(compatibility_disjoint_coefficient*disjoint_nodes))/max_nodes
            
        connection_distance = 0.0
        if me.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in me.connections:
                    disjoint_connections += 1
                        
            for k1,c1 in me.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    connection_distance += c1.distance(c2)
                        
            max_conn = max(len(me.connections),len(other.connections))
            connection_distance = (connection_distance+
                                    (compatibility_disjoint_coefficient*disjoint_connections))/max_conn
                
        distance = node_distance+connection_distance
        return distance
        
    def speciate(self,population,generation):
        unspeciated = set(population)
        new_representatives = {}
        new_members = {}
        for sid,s in self.species.items():
            candidate = []
            for gid in unspeciated:
                g = population[gid]
                d = self.distance(s.representative,g)
                candidate.append((d,g))
                    
            ignored_rdist,new_rep = min(candidates,key = lambda x:x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)
            
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]
                
            candidates = []
            for sid,rid in new_representatives.items():
                rep = population[rid]
                d = self.distance(rep,g)
                if d < compatibility_threshold:
                    candidates.append((d,sid))
                    
            if candidates:
                ignored_sdist,sid = min(candidates,key = lambda x:x[0])
                new_members[sid].append(gid)
            else:
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]
                
        self.genome_to_species = {}
        for sid,rid in new_representatives.items():
            s = self.species.get(sid)
            if s is None:
                s = Species(sid,generation)
                self.species[sid] = s
            
            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid
                
            member_dict = dict((gid,population[gid]) for gid in members)
            s.update(population[rid],member_dict)
            
        def get_species_id(self,indivisual_id):
            return self.genome_to_species[indivisual_id]