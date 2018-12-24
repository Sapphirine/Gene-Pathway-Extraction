'''
Author: Ziyu Chen
'''
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab
import pandas as pd
import numpy as np
import json


def draw_graph(graph_type = 'related', prediction_path = './data/predictions.csv', kegg_genes = './data/kegg_genes.json', 
               all_genes = './data/all_other_genes.json'):
    '''
    Description: Draw connected graph
    graph_type: 'all' for all genes in the corpus; 'related' for knonwn lung cancer genes and related genes; 'known' for already known cancer genes.
    '''
    
    # Load predictions 
    pred_dat = pd.read_csv(prediction_path)
    
    # Load genes
    fp = open('kegg_genes.json', encoding='utf-8')
    kegg_genes_dict = json.load(fp)
    fp.close()
    f = open('all_genes.json', encoding='utf-8')
    all_genes_dict = json.load(f)
    f.close()
    
    kegg_genes = list(kegg_genes_dict.keys())
    all_genes = list(all_genes_dict.keys())
    
    # Convert to pairs
    all_pairs = []
    for i in range(len(pred_dat)):
        if pred_dat.iloc[i,:]['prediction'] != 1:
            all_pairs.append((pred_dat.iloc[i,1], pred_dat.iloc[i,2]))

    related_pairs = []
    for i in range(len(pred_dat)):
        if pred_dat.iloc[i,:]['prediction'] != 1 and (pred_dat.iloc[i,1] in kegg_genes or pred_dat.iloc[i,2] in kegg_genes):
            related_pairs.append((pred_dat.iloc[i,1], pred_dat.iloc[i,2]))

    kegg_pairs = []
    for i in range(len(pred_dat)):
        if pred_dat.iloc[i,:]['prediction'] != 1 and (pred_dat.iloc[i,1] in kegg_genes and pred_dat.iloc[i,2] in kegg_genes):
            kegg_pairs.append((pred_dat.iloc[i,1], pred_dat.iloc[i,2]))

    # List of related genes
    related_genes = []
    for pair in related_pairs:
        for gene in pair:
            if gene not in kegg_genes: 
                related_genes.append(gene)
               
            
    all_genes = list(set(all_genes))
    related_genes = list(set(related_genes))  
    kegg_genes = list(set(kegg_genes))
    
   # Plot graph
   # Some graph parameters  
    G = nx.DiGraph()
    val_map = {'A': 1.0,
               'D': 0.5714285714285714,
               'H': 0.0} 
    
    if graph_type == 'all':
        G.add_edges_from(all_pairs)

        color_map = []
        for node in G:
            if node in kegg_genes:
                color_map.append('blue')
            elif node in related_genes:
                color_map.append('red')
            else:
                color_map.append('pink')

        pos=nx.spring_layout(G)
        node_labels = {node:node for node in G.nodes()}
        plt.figure(1,figsize=(50,50)) 
        nx.draw_networkx_labels(G, pos)
        nx.draw(G,pos, node_color = color_map, node_size=500)
        pylab.show()
    elif graph_type == 'related':
        G.add_edges_from(related_pairs)

        color_map = []
        for node in G:
            if node in kegg_genes:
                color_map.append('blue')
            elif node in related_genes:
                color_map.append('red')

        pos=nx.spring_layout(G)
        node_labels = {node:node for node in G.nodes()}
        plt.figure(1,figsize=(50,50)) 
        nx.draw_networkx_labels(G, pos)
        nx.draw(G,pos, node_color = color_map, node_size=650)
        pylab.show()
    elif graph_type == 'known':
        G.add_edges_from(kegg_pairs)
        color_map = ['blue'] * len(kegg_pairs)

        pos=nx.spring_layout(G)
        node_labels = {node:node for node in G.nodes()}
        plt.figure(1,figsize=(20,20)) 
        nx.draw_networkx_labels(G, pos)
        nx.draw(G,pos, node_color = color_map, node_size=800)
        pylab.show()

 
    