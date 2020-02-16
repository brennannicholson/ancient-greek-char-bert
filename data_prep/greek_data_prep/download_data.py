"""Downloads the Perseus and F1KG files. It should work with other versions of the Perseus and F1KG, but for the sake of reproducibility the specific versions used during development are downloaded."""
import subprocess
import sys


# test whether unzip is installed
def check_unzip():
    try:
        subprocess.run(
            ["unzip"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
    except FileNotFoundError:
        print(
            "unzip is required but not installed. Try installing it with 'sudo apt-get install unzip' and then run this script again.",
            file=sys.stderr,
        )
        sys.exit(1)


def download_git_repo(address, commit):
    repo_name = address.split("/")[-1]
    subprocess.run(["git", "clone", address])
    subprocess.run(["git", "checkout", commit], cwd=repo_name)


def download(address):
    filename = address.split("/")[-1]
    subprocess.run(["wget", address])
    subprocess.run(["unzip", filename])


def get_data():
    check_unzip()
    perseus_repo = "https://github.com/PerseusDL/canonical-greekLit"
    perseus_commit = "803db1425219cd300d907e19c2cb958e6bd5cbbd"
    f1kg_address = "https://zenodo.org/record/2592513/files/OpenGreekAndLatin/First1KGreek-1.1.4529.zip"
    download_git_repo(perseus_repo, perseus_commit)
    download(f1kg_address)


if __name__ == "__main__":
    get_data()
