import os
from os import path

INPUT_PATH = "graphs/matrices"
OUTPUT_PATH = "graphs/output"

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

    if not path.isdir(INPUT_PATH):
        os.mkdir(path.join(os.getcwd(), INPUT_PATH))
    else:
        return

    with open(FILE_PATH) as file:
        graphs = file.read().split("\n\n")
        for i, graph in enumerate(graphs):
            if graph:
                with open(path.join(os.getcwd(), INPUT_PATH, f"graph{i + 1}.txt"), "w") as output:
                    output.write(graph)


if __name__ == "__main__":
    split_graphs(path.join(os.getcwd(), "graphs/list_182_graphs.mat"))
    file_crawler()