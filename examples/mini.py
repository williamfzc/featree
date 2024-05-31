from featree import gen_tree


def main():
    tree = gen_tree()

    # walk
    count = 0

    def counter(*_, **__):
        nonlocal count
        count += 1

    tree.walk_dfs(counter)
    dfs_count = count
    count = 0
    tree.walk_bfs(counter)
    bfs_count = count
    assert dfs_count == bfs_count

    # dump
    node_tree = tree.to_node_tree()
    j = node_tree.model_dump_json()
    with open('mini.json', 'w') as f:
        f.write(j)


if __name__ == "__main__":
    main()
