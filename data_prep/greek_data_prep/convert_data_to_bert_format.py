"""Converts the split data into the format expected by the BERT code."""


def convert_data():
    print("Converting to BERT format...")
    for file_type in ["train", "dev", "test"]:
        with open(f"{file_type}.txt", "r") as input_file:
            data = input_file.read().replace(" ", "_").splitlines()
        with open(f"{file_type}.txt", "w") as output_file:
            for i, line in enumerate(data):
                # replace spaces with underscores, to work around deeply embedded whitespace remove in the BERT implementation
                output_file.write(line + "\n")
                if i % 1000 == 999:
                    # append a new line every 1000 lines (to comply with BERT document formating expectations)
                    output_file.write("\n")


if __name__ == "__main__":
    convert_data()
