import matplotlib.pyplot as plt
from ase.io import read, write
import os
from sys import argv
import networkx as nx
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.potentials.ml_potentials import MachineLearningPotential
from topsearch.plotting.stationary_points import plot_stationary_points
from topsearch.plotting.disconnectivity import plot_disconnectivity_graph
from topsearch.plotting.network import plot_network
from topsearch.plotting.network import barrier_reweighting
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.data.coordinates import MolecularCoordinates


def calc_network_dists(ktn, ktn_ml, ats, thresh=1.5):
    """Calculates distance between the minima of two KTN objects.
    
    Params
    -------------
    ktn:: DFT or reference KineticTransitionNetwork
    ktm_ml:: KTN to compare to (usually and MLP-generated one)
    ats:: reference ase.Atoms object 
            Should be the same ordering of atoms for both sets of coordinates in ktns
    thresh:: distance threshold below which we accept mininima as the same and compare
    
    Returns
    ------------
    dists: list of tuples (i, j, d) where i is the DFT minima index, j is the MLP minima index
    and d is the Cartesian distance between minima at optimal alignment
    
    Notes
    ------------
    Side-effect: lowest-energy minimum from each network is added as min_i and min_e
    properties to the ktns for later use"""
    
    
    comparer = MolecularSimilarity(distance_criterion=1.0, # these settings are not used
                               energy_criterion=5e-3,
                               weighted=False,
                               allow_inversion=True)
    
    
    dists = []
    for i in range(ktn.n_minima):
        print(f"Working on {i}")
        min_coords = MolecularCoordinates(ats.get_chemical_symbols(), 
                                          ktn.get_minimum_coords(i))
        current_dist = 1e6
        curi = 0; curj=0
        for j in range(ktn_ml.n_minima):
            c = comparer.closest_distance(min_coords,
                                          ktn_ml.get_minimum_coords(j))
            if c < current_dist:
                print(f'found new closest {j} at distance {c:6.3f}')
                current_dist = c
                curi = i; curj = j
        if current_dist < thresh:
            dists.append((curi,curj,current_dist))
        
    min_i = np.argmin([ktn.G.nodes[i]["energy"] for i in [j[0] for j in dists]])
    min_e = ktn.G.nodes[min_i]["energy"]
    print('min_e is at ', min_i, "with e ", min_e)
    ktn.min_i = min_i
    ktn.min_e = min_e
    min_i_ml = dists[min_i][1]
    min_e_ml = ktn_ml.G.nodes[min_i_ml]["energy"]
    ktn_ml.min_i = min_i_ml
    ktn_ml.min_e = min_e_ml
        
    return dists


def calc_network_dists_ts(ktn, ktn_ml, ats, dists, thresh=1.5):
    """Calculates distance between the transition states of two KTN objects.
    
    Params
    -------------
    ktn:: DFT or reference KineticTransitionNetwork
    ktm_ml:: KTN to compare to (usually and MLP-generated one)
    ats:: reference ase.Atoms object
    dists:: the list of tuples generated by calc_network_dists
    thresh:: distance threshold to accept TS as the same
    
    Returns
    ------------
    ts_dists: list of tuples (i, j, d) like calc_network_dists
    ts_es: energy difference between TS, shifted by the energy of the lowest minimum in the ktn with each method
    
    Notes
    ------------
    """
    comparer = MolecularSimilarity(distance_criterion=1.0,
                               energy_criterion=5e-3,
                               weighted=False,
                               allow_inversion=True)
    
    mapping = {i[0]: i[1] for i in dists}
    pl = list(ktn.G.edges)
    
    ts_dists = []
    ts_es = []
    for i in range(ktn.n_ts):
        print(f"Working on {i}")
        ts_coords = MolecularCoordinates(ats.get_chemical_symbols(), 
                                         ktn.get_ts_coords(*pl[i]))
        try:
            pli, plj = mapping[pl[i][0]], mapping[pl[i][1]]
        except:
            print("No mapping between ", pl[i][0], pl[i][1])
            continue
        if ktn_ml.G.has_edge(pli, plj):
            c = comparer.closest_distance(ts_coords, ktn_ml.get_ts_coords(pli, plj))
            if c < thresh:
                ts_dists.append(c)
                ts_es.append(ktn.get_ts_energy(*pl[i])-ktn.min_e - \
                    ktn_ml.get_ts_energy(pli, plj)+ktn_ml.min_e)
        
    return ts_dists, ts_es

def canonicalise(f, ktn):
    """Use RDKit to define canonical ordering of atoms in case ktn and ktn_ml have different orders"""
    m1 = Chem.MolFromXYZFile(f)
    rdDetermineBonds.DetermineConnectivity(m1)
    m1_neworder = np.array(tuple(zip(
        *sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(m1))])
        )))[1]
    ats = read(f)
    ats = ats[m1_neworder]
    
    for i in range(ktn.n_minima):
        ktn.G.nodes[i]["coords"] = ktn.G.nodes[i]["coords"].reshape(-1, 3)[m1_neworder].flatten()
        
    for i,j in ktn.G.edges():
        ktn.G[i][j]["coords"] = ktn.G[i][j]["coords"].reshape(-1, 3)[m1_neworder].flatten()
    
    return ats


def plot_network_diff(ktn, ktn_ml, dists, vs=[-0.01, 0.01], seed=42, sf=2, thresh=1.0):
        """Plots the overlaid and compared graphs of KTNs"""
        fig, axs = plt.subplots()
        colour_scheme='cool'

        g_weighted = barrier_reweighting(ktn)
        pos = nx.spring_layout(g_weighted, seed=seed)
        colours = np.empty((0))

        ktn.G.remove_edges_from(nx.selfloop_edges(ktn.G))
        for i in ktn.G.nodes:
                colours = np.append(colours, ktn.G.nodes[i]['energy']-ktn.G.nodes[0]['energy'])
        print(colours)

        network_contours = nx.draw_networkx_nodes(
                ktn.G, pos, node_color=colours,
                cmap=plt.get_cmap(colour_scheme), ax=axs, vmin=vs[0], vmax=vs[1])


        nx.draw_networkx_edges(ktn.G, pos)
        cb = plt.colorbar(network_contours, label='Energy difference (eV)')
        np.random.seed(seed)
        pos_ml = {}
        skips=[]
        for i in dists:
                if i[2] > thresh:
                        skips.append(i[0])
                        continue
                d = np.random.rand(2)
                d = d/np.linalg.norm(d) * i[2] * sf
                de = ktn_ml.G.nodes[i[1]]['energy'] - ktn_ml.min_e + ktn.min_e - ktn_ml.G.nodes[dists[0][1]]['energy']
                print(de)
                new_pos = [[pos[i[0]][0]+d[0]], [pos[i[0]][1]+d[1]]]
                pos_ml[i[1]] = new_pos
                axs.plot([new_pos[0][0], pos[i[0]][0]], [new_pos[1][0], pos[i[0]][1]], ls='-', c='tab:gray', zorder=0.1)
                axs.scatter(*new_pos, s=50, c=de, cmap=network_contours.cmap, vmin=vs[0], vmax=vs[1], zorder=10, edgecolors='k')

        for (i,j) in ktn.G.edges():
                print(i,j)
                if i in skips or j in skips:
                        print("skipping", i, j)
                        continue    
                i_n = dists[i][1]
                j_n = dists[j][1]
                if ktn_ml.G.has_edge(dists[i][1], dists[j][1]):
                        c='k'
                        print("yes",i_n, j_n)
                else:
                        c='r'
                print(i_n, j_n)
                print(pos_ml[i_n])
                print(pos_ml[j_n])
                axs.plot([pos_ml[i_n][0], pos_ml[j_n][0]], [pos_ml[i_n][1], pos_ml[j_n][1]], c=c, ls='--', zorder=0.01)
        
        return fig, axs
    

# calculate the missing ones and find their nearest too
def calc_network_dists_missing(ktn, ktn_ml, ats, sel1=[], sel2=[]):
    
    comparer = MolecularSimilarity(distance_criterion=1.0,
                               energy_criterion=5e-3,
                               weighted=False,
                               allow_inversion=True)
    
    dists = []
    for j in range(ktn_ml.n_minima):
        if len(sel2) and j not in sel2:
            continue
        print(f"Working on {j}")
        min_coords = MolecularCoordinates(ats.get_chemical_symbols(), ktn_ml.get_minimum_coords(j))
        current_dist = 1e6
        curi = 0; curj=0
        for i in range(ktn.n_minima):
            if len(sel1) and i not in sel1:
                continue
            c = comparer.closest_distance(min_coords, ktn.get_minimum_coords(i))
            if c < current_dist:
                print('found with distance ', c)
                current_dist = c
                curi = i; curj = j
        dists.append((curi,curj,current_dist))
        
    return dists
    
def remove_inversions(ktn, ats):
    """Removes minima related by inversion symmetry, choosing the version with the fewest connected TS to remove"""
    to_del = []
    to_del2 = []
    comparer = MolecularSimilarity(distance_criterion=0.5,
                        energy_criterion=1e-3,
                        weighted=False,
                        allow_inversion=True)
    for j in range(ktn.n_minima):
        if j in to_del:
            continue
        min_coords = MolecularCoordinates(ats.get_chemical_symbols(), ktn.get_minimum_coords(j))
        for i in range(j+1, ktn.n_minima):
            if not np.isclose(ktn.get_minimum_energy(j), ktn.get_minimum_energy(i), atol=5e-3):
                continue
            if comparer.test_same(min_coords, ktn.get_minimum_coords(i), 
                               ktn.get_minimum_energy(j), ktn.get_minimum_energy(i)):
                to_del.append(i)
                to_del2.append(j)
                
    print(to_del, to_del2)
    # then decide to remove depending on how many edges each contains
    edges = list(ktn.G.edges())

    to_del_master = []
    for i in range(len(to_del)):
        nedge1 = sum([1 if to_del[i] in j else 0 for j in edges])
        nedge2 = sum([1 if to_del2[i] in j else 0 for j in edges])
        if nedge2 > nedge1:
            to_del_master.append(to_del[i])
            print('removed ', to_del[i])
        else:
            to_del_master.append(to_del2[i])
            print('removed ', to_del2[i])
    
    for i in reversed(sorted(to_del_master)):
        ktn.remove_minimum(i)
        
        
def calc_table(ktn, ktn_ml, ats, dists, thresh=1.0):
    """"Calcuates values needed for aggregated statistics for the table"""
    d_m = [i[2]**2 for i in dists]
    # d_m = [i[2]**2 for i in dists if i[2]<thresh]
    d_ts, de_ts = calc_network_dists_ts(ktn, ktn_ml, ats, dists, thresh=thresh)
    de_m = [ktn.get_minimum_energy(i[0])-ktn.min_e - \
        ktn_ml.get_minimum_energy(i[1])+ktn_ml.min_e for i in dists]
    tm = ktn.n_ts - len(d_ts)
    tp = ktn_ml.n_ts - len(d_ts)
    mm  = ktn.n_minima - len(d_m) 
    mp = ktn_ml.n_minima - len(d_m)
    
    return d_m, d_ts, de_m, de_ts, mp, mm, tp, tm



if __name__ == '__main__':
    mol = argv[1]
    model = argv[2]
    try:
        extra = argv[3]
    except:
        extra = ""
    owd = os.getcwd()

    # load DFT network
    ktn = KineticTransitionNetwork()
    os.chdir(f"/dccstor/aimft1/landscape_paper/dft/{mol}")
    ktn.read_network()
    ats = canonicalise(f"{mol}.xyz", ktn)
    # remove self-edges
    for i in list(nx.selfloop_edges(ktn.G)):
        ktn.remove_ts(*i)
    to_del = []
    for i in ktn.G.nodes:
        if ktn.G.nodes[i] == {}:
            to_del.append(i)
    for i in reversed(to_del):
        ktn.G.remove_node(i)
    ktn.n_ts = len(list(ktn.G.edges))
    print('dft: ', ktn.n_minima, " mins, ", ktn.n_ts, " TS")
    
    # load MLP network
    ktn_ml = KineticTransitionNetwork()
    os.chdir(f"/dccstor/aimft1/landscape_paper/{model}/{mol}")
    ktn_ml.read_network()
    canonicalise(f'/dccstor/aimft1/landscape_paper/molecules_relax/{mol}_relax.xyz', ktn_ml)
    print("ml before inversions: ", ktn_ml.n_minima, ktn_ml.n_ts)
    remove_inversions(ktn_ml, ats)
    print("ml after inversions: ", ktn_ml.n_minima, ktn_ml.n_ts)
    print('--------------------', flush=True)
    
    
    print("doing dists")
    # calc distances when aligned to DFT minima
    dists = calc_network_dists(ktn, ktn_ml, ats)
    print(dists)
    print('--------------------')

    missing = [i for i in range(ktn_ml.n_minima) if i not in [j[1] for j in dists]]
    print("Extra ML minima: ", missing)
    if len(missing):
        dists_2 = calc_network_dists_missing(ktn, ktn_ml, ats, sel2=missing)
    else:
        dists_2 = []
    print('--------------------')

        
    dat = calc_table(ktn, ktn_ml, ats, dists)

    print('saving')
    os.chdir(owd)
    with open(f"analysis_{model}_{mol}{extra}.pkl", 'wb') as f:
        pickle.dump([ktn, ktn_ml, dists, dists_2, dat], f) 
        