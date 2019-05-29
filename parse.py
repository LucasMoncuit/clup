import argparse
import sys

from gat import main
from gat.main import read_graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRP Graph Toolkit");
    parser.add_argument("--normalize", action="store_true");
    parser.add_argument("--full", action="store_true");
    parser.add_argument("--reify", action="store_true");
    parser.add_argument("--format");
    parser.add_argument("input", nargs="?",
                        type=argparse.FileType("r"), default=sys.stdin);
    parser.add_argument("output", nargs="?",
                        type=argparse.FileType("w"), default=sys.stdout);
    arguments = parser.parse_args();

    graphs = read_graphs(arguments.input, format='mrp',
                         full=arguments.full, normalize=arguments.normalize,
                         reify=arguments.reify);

    for e in graphs:
        print(e.encode())
