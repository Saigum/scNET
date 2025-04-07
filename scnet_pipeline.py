#!/usr/bin/env python3
"""
scnet_pipeline.py

Convert the scNET notebook into a runnable script. Downloads example data, runs scNET with
specified parameters for a P100 GPU, and saves all results to files.
"""
import os
import gdown
import scanpy as sc
import scNET
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc


def calculate_marker_gene_aupr(adata, marker_genes, cell_types, output_path):
    """
    Calculate and save Precision-Recall AUPR curves for given marker genes.
    """
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette(n_colors=len(marker_genes))

    for marker_gene, cell_type, color in zip(marker_genes, cell_types, colors):
        expr = adata[:, marker_gene].X.toarray().flatten()
        labels = adata.obs["Cell Type"].isin(cell_type).astype(int)
        precision, recall, _ = precision_recall_curve(labels, expr)
        aupr = auc(recall, precision)
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'PRAUC={aupr:.2f} for {marker_gene} ({cell_type[0]})')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve by Cell Type', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    # 1) Download example data
    url = 'https://drive.google.com/uc?id=1C_G14cWk95FaDXuXoRw-caY29DlR9CPi'
    input_h5ad = 'example.h5ad'
    if not os.path.exists(input_h5ad):
        gdown.download(url, input_h5ad, quiet=False)

    # 2) Configure scNET for P100 GPU (16–12 GB VRAM)
    scNET.main.MAX_CELLS_BATCH_SIZE = 2000  # reduce from default for P100
    num_batches = 200

    # 3) Run scNET on full dataset
    obj = sc.read_h5ad(input_h5ad)
    scNET.run_scNET(
        obj,
        pre_processing_flag=False,
        human_flag=False,
        number_of_batches=num_batches,
        split_cells=True,
        max_epoch=300,
        model_name='test'
    )

    # 4) Load and save embeddings
    eg, ec, nf, of = scNET.load_embeddings('test')
    np.savez(
        'embeddings_test.npz',
        embedded_genes=eg,
        embedded_cells=ec,
        node_features=nf,
        out_features=of
    )

    # 5) Reconstruct AnnData object and save
    cell_types_map = {
        "0": "Macrophages", "1": "Macrophages", "2": "CD8 Tcells",
        "3": "Microglia",   "4": "Cancer",     "5": "CD4 Tcells",
        "6": "B Cells",     "8": "Cancer",     "10": "Prolifrating Tcells",
        "11": "NK"
    }
    obj.obs['Cell Type'] = obj.obs.seurat_clusters.map(cell_types_map)
    recon = scNET.create_reconstructed_obj(nf, of, obj)
    recon.write_h5ad('reconstructed_test.h5ad')

    # 6) Build co-embedded network, save graph and modularity
    net, mod = scNET.build_co_embeded_network(eg, nf)
    nx.write_gpickle(net, 'co_embedded_network_test.gpickle')
    with open('modularity_test.txt', 'w') as f:
        f.write(str(mod))

    # 7) Marker gene AUPR
    markers = ['Cd8a', 'Cd4', 'Cd14', 'P2ry12', 'Ncr1', 'Mki67', 'Tert']
    types_list = [
        ['CD8 Tcells'], ['CD4 Tcells'], ['Macrophages'],
        ['Microglia'],    ['NK'],         ['Prolifrating Tcells'],
        ['Cancer']
    ]
    calculate_marker_gene_aupr(recon, markers, types_list, 'marker_aupr.png')

    # 8) Signature propagation
    scNET.run_signature(recon, up_sig=["Zap70","Lck","Fyn","Cd3g","Cd28","Lat"], alpha=0.9)
    scNET.run_signature(recon, up_sig=["Cdkn2a","Myc","Pten","Kras"])

    # 9) Re-embed T-cell subset
    sub = obj[obj.obs['Cell Type'] == 'CD8 Tcells'].copy()
    scNET.run_scNET(
        sub,
        pre_processing_flag=False,
        human_flag=False,
        number_of_batches=num_batches,
        split_cells=False,
        max_epoch=300,
        model_name='Tcells'
    )
    eg2, ec2, nf2, of2 = scNET.load_embeddings('Tcells')
    net2, mod2 = scNET.build_co_embeded_network(eg2, nf2, 99.5)
    nx.write_gpickle(net2, 'co_embedded_network_Tcells.gpickle')
    with open('modularity_Tcells.txt', 'w') as f:
        f.write(str(mod2))

    # 10) Downstream TF scoring and plot
    tf_scores = scNET.find_downstream_tfs(net2, ["Zap70","Lck","Fyn","Cd3g","Cd28","Lat"]) \
        .sort_values(ascending=False).head(10)
    tf_scores.to_csv('tf_scores.csv')
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=tf_scores.index, y=tf_scores.values, color='skyblue')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('tf_scores.png')
    plt.close()

    # 11) Pathway enrichment and plot
    subset = recon[recon.obs['Cell Type'].isin(['Microglia','Macrophages','Cancer'])].copy()
    de_grp, sig_path, filt_kegg, enr_res = scNET.pathway_enricment(subset, groupby='Cell Type')
    sig_path.to_csv('significant_pathways.csv')
    enr_res.to_csv('enrichment_results.csv')
    scNET.plot_de_pathways(sig_path, enr_res, 10)
    plt.savefig('de_pathways.png')
    plt.close()


if __name__ == '__main__':
    main()
