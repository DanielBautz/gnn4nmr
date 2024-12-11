# playground.py

import argparse
from tools import load_graphs, analyze_graph, convert_xyz_to_sdf

def main():
    parser = argparse.ArgumentParser(description="Playground for running tools.")

    subparsers = parser.add_subparsers(dest="command")

    # Subparser for loading and analyzing graphs
    graph_parser = subparsers.add_parser("analyze_graph", help="Analyze graphs from a pickle file.")
    graph_parser.add_argument("file_path", type=str, help="Path to the pickle file containing graphs.")

    # Subparser for converting XYZ to SDF
    convert_parser = subparsers.add_parser("convert_xyz_to_sdf", help="Convert XYZ files to SDF format.")
    convert_parser.add_argument("base_dir", type=str, help="Base directory containing XYZ files.")
    convert_parser.add_argument("--output_subdir", type=str, default="converted_sdf_files", help="Subdirectory for converted SDF files.")
    convert_parser.add_argument("--start", type=int, default=1, help="Starting directory index.")
    convert_parser.add_argument("--end", type=int, default=100, help="Ending directory index.")

    args = parser.parse_args()

    if args.command == "analyze_graph":
        graphs = load_graphs(args.file_path)
        if isinstance(graphs, list):
            for i, graph in enumerate(graphs):
                print(f"\n--- Graph {i + 1} ---")
                analysis = analyze_graph(graph)
                print(analysis)
        elif isinstance(graphs, nx.Graph):
            analysis = analyze_graph(graphs)
            print(analysis)
        else:
            print("Unexpected format: The file does not contain valid graphs.")

    elif args.command == "convert_xyz_to_sdf":
        results = convert_xyz_to_sdf(args.base_dir, args.output_subdir, args.start, args.end)
        for result in results:
            print(result)

if __name__ == "__main__":
    main()
