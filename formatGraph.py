import os
from os import path

INPUT_PATH = "graphs/matrices"
OUTPUT_PATH = "graphs/graph_data"

def convert_to_edge_list(adj_mat):
  rows = adj_mat.read().split('\n')
  edge_list = []

  for i, row in enumerate(rows):
    v1 = i + 1

    nodes = row.split()
    for v2, connection in enumerate(nodes[i+1:]):
      if connection == "1":
        v2 = i + 1 + v2 + 1
        edge_list.append(f"{v1} {v2}")
  
  return edge_list

def file_crawler():
    if not path.isdir(OUTPUT_PATH):
        os.mkdir(path.join(os.getcwd(), OUTPUT_PATH))
    else:
        return
    
    for i, filename in enumerate(os.listdir(path.join(os.getcwd(), INPUT_PATH))):
        with open(path.join(os.getcwd(), INPUT_PATH, filename)) as adj_mat:
            with open(path.join(os.getcwd(), OUTPUT_PATH, f"graph{i + 1}.txt"), "w") as output:
                output.write("\n".join(convert_to_edge_list(adj_mat)))

def split_graphs(FILE_PATH):

    if not path.isdir(path.join(os.getcwd(), OUTPUT_PATH)):
        os.mkdir(path.join(os.getcwd(), OUTPUT_PATH))

    with open(FILE_PATH) as file:
        graphs = file.read().split("\n\n\n\n")
        for i, graph in enumerate(graphs):
            if graph:

                adj_list, graph_data = graph.split("\n\n")
                mds = graph_data[graph_data.find("Minimum Dominating Set: ") + len("Minimum Dominating Set: ")]

                lists = adj_list.split("\n")

                edge_list = []

                for node in lists:
                    node_split = node.split(": ")

                    if len(node_split) > 1:
                        node, neighbors = node_split
                    else:
                        edge_list.append(f"{node[:-1]} {node[:-1]}")
                        continue

                    neighbors = neighbors.split(" ")

                    node = int(node)
                    for neighbor in neighbors:
                        neighbor = int(neighbor)

                        if neighbor > node:
                            edge_list.append(f"{node} {neighbor}")

                with open(path.join(os.getcwd(), OUTPUT_PATH, f"graph{(i + 1):04}.txt"), "w") as output:
                    output.write(f"{mds}\n")
                    output.write("\n".join(edge_list))


if __name__ == "__main__":
    split_graphs(path.join(os.getcwd(), "graphs/list_182_graphs.txt"))