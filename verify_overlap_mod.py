import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rasterio
from fct_data_extraction import little_pair


def verify_overlap_condition(path_centerline, batch_size, scale, overlap_pct, visualize=True):

    # Création de l'objet little_pair
    lp = little_pair(path_centerline, batch_size, scale, 
                     overlap=1, percentage_overlap=overlap_pct)
    
    # Récupération les positions des centres 
    r_positions = np.array(lp.r_array)
    c_positions = np.array(lp.c_array)
    n_total = len(r_positions)
    print(f"Nombre total de positions valides : {n_total}")
    
    
    min_distance = (batch_size * (1 - overlap_pct)) ** 2
    
    
    selected_indices = []
    
    for i in range(n_total):
        overlap = False
        for j in selected_indices:
            dr = r_positions[i] - r_positions[j]
            dc = c_positions[i] - c_positions[j]
            # Distance euclidienne au carré
            dist = dr**2 + dc**2
            if dist < min_distance:
                overlap = True
                break
        if not overlap:
            selected_indices.append(i)
    
    n_selected = len(selected_indices)
    print(f"Positions retenues après selection (distance euclidienne) : {n_selected} sur {n_total}")
    
    selected_coords = np.column_stack((r_positions[selected_indices], c_positions[selected_indices]))
    return selected_coords

        


def visualize_comparison(path_centerline, coords_DE, coords_real, batch_size, overlap_pct):
    with rasterio.open(path_centerline) as src:
        img = src.read(1)
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(img, cmap='gray', alpha=0.4, origin='upper')
    
    half = batch_size / 2
    
    # Rectangles BLEUS pour les crd DE
    for i, (r, c) in enumerate(coords_DE):
        rect_de = Rectangle((c - half, r - half), batch_size, batch_size,
                             linewidth=2, edgecolor='blue', facecolor='none', 
                             linestyle='-', alpha=0.6, 
                             label=' (DE)' if i == 0 else "")
        ax.add_patch(rect_de)
        # Petit marqueur bleu pour le centre
        ax.plot(c, r, 'b+', markersize=4, alpha=0.6)

    # 3. Dessiner les patches RÉELS (Crop) -> Rectangles ROUGES
    # On utilise unique pour ne pas superposer 10 fois le même rectangle si plusieurs dates
    coords_real_unique = np.unique(coords_real, axis=0)
    
    for i, (r, c) in enumerate(coords_real_unique):
        rect_real = Rectangle((c - half, r - half), batch_size, batch_size,
                              linewidth=1.5, edgecolor='red', facecolor='none', 
                              linestyle='--', alpha=0.8, 
                              label=f'Réel (Crop) - {len(coords_real_unique)} pos' if i == 0 else "")
        ax.add_patch(rect_real)
        
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()