def write_to_file(filename, data):
    with open(filename, "w") as f:
        for text in data:
            f.write("%s\n" % text)