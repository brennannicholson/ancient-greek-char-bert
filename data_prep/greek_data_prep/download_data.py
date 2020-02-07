"""Downloads the Perseus and F1KG files. It should work with other versions of the Perseus and F1KG, but for the sake of reproducibility the specific versions used during development are downloaded."""
import subprocess


def download_git_repo(address, commit):
    repo_name = address.split("/")[-1]
    subprocess.run(["git", "clone", address])
    subprocess.run(["git", "checkout", commit], cwd=repo_name)


def download(address):
    filename = address.split("/")[-1]
    subprocess.run(["wget", address])
    subprocess.run(["unzip", filename])


def get_data():
    perseus_repo = "https://github.com/PerseusDL/canonical-greekLit"
    perseus_commit = "803db1425219cd300d907e19c2cb958e6bd5cbbd"
    f1kg_address = "https://zenodo.org/record/2592513/files/OpenGreekAndLatin/First1KGreek-1.1.4529.zip"
    download_git_repo(perseus_repo, perseus_commit)
    download(f1kg_address)


if __name__ == "__main__":
    get_data()
