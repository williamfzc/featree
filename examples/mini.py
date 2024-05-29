from featree import gen_tree


def main():
    tree = gen_tree()
    node_tree = tree.to_node_tree()
    j = node_tree.model_dump_json()
    print(j)


if __name__ == "__main__":
    main()
