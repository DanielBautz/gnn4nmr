import os
import pickle
import networkx as nx
from openbabel import pybel

def load_graphs(file_path):
    """Lädt Graphen aus einer Pickle-Datei."""
    with open(file_path, 'rb') as file:
        graphs = pickle.load(file)
    return graphs

def analyze_graph(graph):
    """Analysiert einen Graphen und gibt Eigenschaften aus."""
    analysis = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        "is_connected": nx.is_connected(graph),
    }
    if analysis["is_connected"]:
        analysis["diameter"] = nx.diameter(graph)
    else:
        analysis["connected_components"] = nx.number_connected_components(graph)
    return analysis

def convert_xyz_to_sdf(base_dir, output_subdir="converted_sdf_files", start=1, end=100):
    """Konvertiert XYZ-Dateien in SDF-Format."""
    output_dir = os.path.join(base_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i in range(start, end + 1):
        dir_name = f"{i:03d}"
        full_dir_path = os.path.join(base_dir, dir_name)

        if not os.path.isdir(full_dir_path):
            results.append(f"Directory {full_dir_path} does not exist, skipping...")
            continue

        for j in range(10):
            xyz_filename = f"{dir_name}_{j:02d}.xyz"
            xyz_path = os.path.join(full_dir_path, xyz_filename)

            if not os.path.isfile(xyz_path):
                results.append(f"File {xyz_path} not found, skipping...")
                continue

            sdf_filename = f"{dir_name}_{j:02d}.sdf"
            sdf_path = os.path.join(output_dir, sdf_filename)

            try:
                mol = next(pybel.readfile("xyz", xyz_path))
                mol.write("sdf", sdf_path, overwrite=True)
                results.append(f"Converted: {xyz_path} -> {sdf_path}")
            except StopIteration:
                results.append(f"Could not read {xyz_path}. File might be empty or corrupted.")
            except Exception as e:
                results.append(f"Error converting {xyz_path}: {e}")

    return results
