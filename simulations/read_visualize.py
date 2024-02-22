import numpy as np
import pandas as pd
from cd_v_partition.utils import edge_to_adj, get_scores
from common_funcs import save
import os
import matplotlib.pyplot as plt

def read_acc_tradeoff_results(shape, nnodes, hierarchical=False):
    num_repeats = shape[0]
    num_comms = shape[1]
    labels = ['serial', 'mod', 'ec', 'causal']
    scores_serial = None
    time_serial = None
    scores = []
    sizes = []
    time = []
    extension = '_hierarchical' if hierarchical else ''
    for l in labels:
        if l == 'serial':
            scores_serial = np.loadtxt("./simulations/acc_tradeoff_scores_{}{}".format(l, extension))
            time_serial = np.loadtxt("./simulations/acc_tradeoff_time_{}{}".format(l, extension))

        else:
            scores.append(np.loadtxt("./simulations/acc_tradeoff_scores_{}{}".format(l, extension)).reshape(shape))
            time.append(np.loadtxt("./simulations/acc_tradeoff_time_{}{}".format(l, extension)))
            sizes.append(np.loadtxt("./simulations/acc_tradeoff_size_{}{}".format(l,extension)))
    fig, axs = plt.subplots(2, sharex=True)
    tpr_ind = -2
    new_dim = num_repeats*num_comms
    markers = ['^', 'o', '*']
    labels = ['non-partitioned', 'disjoint', 'edge_cover', 'expansive_causal']
    for z,s, l, m in zip(sizes, scores, labels[1:], markers):
        axs[0].scatter(z.reshape(new_dim), s[:,:,tpr_ind].reshape(new_dim), label=l, marker=m)
        
    axs[0].scatter([nnodes for _ in np.coarange(num_repeats)], scores_serial[:,tpr_ind], label='no partition', marker='+')
    axs[0].set_ylabel("TPR")


    for z,t, l, m in zip(sizes, time, labels[1:], markers):
        axs[1].scatter(z.reshape(new_dim), t.reshape(new_dim), label=l, marker=m)
    axs[1].scatter([nnodes for _ in np.arange(num_repeats)], time_serial, label='no partition', marker='+')
    axs[1].set_ylabel("Time (s)")
    axs[1].set_xlabel("Size of largest partition")
    plt.legend()
    plt.savefig("./simulations/acc_scaling_tradoff.png")

def read_experiment_csv(path, parameter, sweep, algs, num_repeats, fusion_algs,x_label, nnodes):
    for f in fusion_algs:
        path_data = "{}/{}/{}_".format(path, f, parameter)
        scores = []
        for alg in algs:
            scores_by_alg = np.zeros((num_repeats, len(sweep), 5))
            for i, s in enumerate(sweep):
                for r in np.arange(num_repeats):
                    if parameter == "nnodes":
                        nnodes = s
                    edges_true = pd.read_csv("{}{}/{}/edges_true.csv".format(path_data, s, r), header=0).to_numpy()
                    G_star = edge_to_adj(edges_true, list(np.arange(nnodes)))
                    edges = pd.read_csv("{}{}/{}/edges_{}.csv".format(path_data, s, r, alg), header=0).to_numpy()
                    scores_by_alg[r][i] = get_scores([alg], [edge_to_adj(edges,list(np.arange(nnodes)))],G_star)
            scores.append(scores_by_alg)
        save(path, scores, algs, num_repeats, sweep, x_label, screen=f=='screen_projections', time=False)
                    
def read_experiment_chkpoints(path_dir, sweep, algs, param_name, num_repeats, x_label):
    score_list = []
    for alg in algs:
        if alg == 'pef':
            path="{}_fix/{}/fusion".format(path_dir, alg)
            print(path)
        else:
            path="{}/{}/screen_projections".format(path_dir, alg)
        scores = np.zeros((num_repeats, len(sweep), 6))
        for i,p in enumerate(sweep):
            if os.path.exists("{}/{}_{}".format(path,param_name, p)):
                for j in range(num_repeats):
                    if os.path.exists("{}/{}_{}/{}".format(path,param_name, p,j)):
                        if alg == 'pef':
                            print(p)
                        if os.path.exists("{}/{}_{}/{}/time_chkpoint.txt".format(path,param_name, p,j)):
                            print("fetching checkpoint")
                            scores[j][i] = np.loadtxt("{}/{}_{}/{}/time_chkpoint.txt".format(path,param_name,p,j))
        score_list.append(scores)
    new_algs = ['no partition','pef', 'edge_cover', 'expansive_causal', 'disjoint']
    save(path_dir, score_list, new_algs, num_repeats, sweep, x_label, screen=True, time=True, remove_incomplete=True, plot_dir=path_dir)

def read_experiment_score_txt(path_dir, sweep, algs, num_repeats, x_label):
    scores_list = []
    for alg in algs:
        if alg == 'pef':
            path="{}_fix/screen_projections".format(path_dir)
        else:
            path="{}/screen_projections".format(path_dir)
        scores = np.loadtxt("{}/scores_{}.txt".format(path, alg)).reshape((num_repeats, len(sweep), -1))
        scores_list.append(scores)
    new_algs = ['no partition','pef', 'edge_cover', 'expansive_causal', 'disjoint']
    save(path_dir, scores_list, new_algs, num_repeats, sweep, x_label, screen=True, time=True, remove_incomplete=True, plot_dir=path_dir)
    
    
def read_experiment_6(path_dir, algs, net_id):
    df = pd.DataFrame(columns=["Algorithm","SHD", "TPR", "FPR","Time (s)"])
    for alg in algs:
        path="{}/{}/screen_projections".format(path_dir, alg)
        print(path)
        if os.path.exists("{}/net_{}".format(path, net_id)):
            if os.path.exists("{}/net_{}/time_chkpoint.txt".format(path, net_id)):
                        print("fetching checkpoint")
                        scores = np.loadtxt("{}/net_{}/time_chkpoint.txt".format(path, net_id))
                        df.loc[len(df.index)] = [alg, scores[0], scores[-3], scores[-2], scores[-1]]
    

    print(df)
    net0 = np.loadtxt("./datasets/bionetworks/ecoli/synthetic_copies/net_{}.txt".format(net_id))
    print(net0.shape, np.sum(net0))
  
#read_experiment_chkpoints("./simulations/experiment_1", [10**i for i in range(1,7)], ['serial','pef', 'edge_cover', 'expansive_causal', 'mod'], 'samples', 30, "Number of samples")
#read_experiment_chkpoints("./simulations/experiment_2", [0, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5], ['serial','pef', 'edge_cover', 'expansive_causal', 'mod'], 'rho', 10, "Rho")
read_experiment_score_txt("./simulations/experiment_3", np.arange(0,4,0.5), ['serial','pef', 'edge_cover', 'expansive_causal', 'mod'], 50, "Fraction of extraneous edges")
read_experiment_score_txt("./simulations/experiment_4",np.arange(0.1, 1, 0.1) , ['serial','pef', 'edge_cover', 'expansive_causal', 'mod'], 50, "Alpha")
#read_experiment_chkpoints("./simulations/experiment_5", [10**i for i in range(1,5)], ['serial','pef', 'edge_cover', 'expansive_causal', 'mod'],'nnodes', 5, "Number of nodes")

#read_experiment_chkpoints("./simulations/experiment_5_fix_comm_3", [10**i for i in range(1,5)], ['serial','pef', 'edge_cover', 'expansive_causal', 'mod'],'nnodes', 5, "Number of nodes")
#read_experiment_6("./simulations/experiment_6_no_comm", ['serial','pef', 'edge_cover', 'expansive_causal', 'mod'], net_id=1)
#read_experiment_6("./simulations/experiment_7", ['serial','pef', 'edge_cover', 'expansive_causal', 'mod'], net_id=0)

# read_acc_tradeoff_results((5,5,5),1000)