from interpreter import read_genome, get_ordered_keys
from random import random, randint

ADD_NODE = .2
DELETE_CONN = .2
ADD_CONN = .2

def evolve(genome):
    did_something = False
    new_genome = {'input': genome['input'], 'output': genome['output'], 'connections': {}}
    node_names = [-1 - r for r in range(genome['output'])]
    node_names = node_names + range(genome['input']) + genome['connections'].keys()
    node_names = list(set(node_names))
    node_names.sort()
    next_node_name = node_names[-1] + 1
    existing_keys = get_ordered_keys(genome)
    replacements = {}
    for k in existing_keys:
        inputs = [c for c in genome['connections'][k]]
        if random() < ADD_CONN:
            did_something = True
            possible_idx = range(0, k) if k >= 0 else range(0, next_node_name)
            possible_idx = [i for i in possible_idx if i not in inputs]
            if len(possible_idx) > 0:
                idx_to_add = randint(0, len(possible_idx) - 1)
                inputs.append(possible_idx[idx_to_add])
        if len(inputs) > 1 and random() < DELETE_CONN:
            did_something = True
            idx_to_delete = randint(0, len(inputs) - 1)
            del inputs[idx_to_delete]
        if random() < ADD_NODE:
            did_something = True
            if k >= 0:
                new_genome['connections'][next_node_name] = [k]
                replacements[k] = next_node_name
            else:
                new_genome['connections'][next_node_name] = inputs
                inputs = [next_node_name]
            next_node_name += 1
        new_genome['connections'][k] = list(set([replacements[i] if (i in replacements and (k < 0 or replacements[i] < k)) else i for i in inputs]))
    if not did_something:
        return evolve(genome)
    return new_genome

if __name__ == '__main__':
    genome = read_genome('genome.txt')
    for i in range(10):
        genome = evolve(genome)
        print(genome)
