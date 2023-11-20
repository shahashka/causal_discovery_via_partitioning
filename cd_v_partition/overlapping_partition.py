import numpy as np
import subprocess


def oslom_algorithm(nodes, data_dir, oslom_dir, structure_type='dag'):
    """Overlapping partitioning methods which take an input graph (superstructure) and partition nodes according to an objective
       overlapping nodes ideally render the partitions conditionally independent

    Args:
        data_dir (str): the directory containing the *.dat file which holds the edges of the structure to partition
        oslom_dir (str): the directory containing the OSLOM binary 
        structure_type (str, optional): specify the structure type as either the 'dag', 
                                        'superstructure', or 'superstructure_weighted'. If weighted
                                        then weights in the *.dat are used by OSLOM. Defaults to 'dag'.

    Returns:
        dict: the estimated partition as a dictionary {comm_id : [nodes]}
    """
    # Run the OSLOM code externally
    weight_flag = "-w" if "weight" in structure_type else "-uw"
    subprocess.run(["{}/oslom_undir".format(oslom_dir), "-f", 
                    "{}/edges_{}.dat".format(data_dir, structure_type),"{}".format(weight_flag)])
    
    # Read the output partition file and return the partition as a dictionary 
    partition_file = "{}/edges_{}.dat_oslo_files/tp".format(data_dir, structure_type)
    with open(partition_file, 'rb') as f:
        lines = f.readlines()
    lines = lines[1::2]
    lines = [[int(node) for node in l.split()] for l in lines]
    partition = dict(zip(np.arange(len(lines)), lines))
    homeless_nodes = list(nodes)
    for part in lines:
        for n in part:
            if n in homeless_nodes:
                homeless_nodes.remove(n)
    if len(homeless_nodes) > 0:
        partition[len(lines)] = homeless_nodes
    return partition
        
            


def partition_problem():
    print("TODO")