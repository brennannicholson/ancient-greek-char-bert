"""Parses (if needed) and cleans F1KG text files and Perseus Cannonical Greek Lit XML files for use with charater-based neural networks. Assumes the files have already been downloaded (if not, run download_data.py first)."""
import glob
import os
import re
from cltk.corpus.utils.formatter import cltk_normalize
from bs4 import BeautifulSoup
from greek_data_prep.utils import write_to_file

# exclude Bacchylides' Odes due to the fragmentary nature of the text
BACHCHYLIDES_ODES = [
    "tlg0199.tlg001.perseus-grc1.xml",
    "tlg0199.tlg002.perseus-grc1.xml",
]

TEXTS_WITH_SIGNIFICANT_OCR_ERRORS = [
    "tlg3129.ogl001.1st1K-grc1.xml",  # In Cyrilli In XII Prophetas Theophylacti
]

# ideally these should be automatically identified and converted but CLTK's parser doesn't seem to work...
BETA_CODE_FILES = [
    "tlg2003.tlg002.perseus-grc1.xml",
    "tlg2003.tlg010.perseus-grc1.xml",
    "tlg2003.tlg007.perseus-grc1.xml",
    "tlg2003.tlg009.perseus-grc1.xml",
    "tlg2003.tlg004.perseus-grc1.xml",
    "tlg2003.tlg012.perseus-grc1.xml",
    "tlg2003.tlg008.perseus-grc1.xml",
    "tlg2003.tlg005.perseus-grc1.xml",
    "tlg2003.tlg011.perseus-grc1.xml",
    "tlg2003.tlg001.perseus-grc1.xml",
    "tlg2003.tlg003.perseus-grc1.xml",
    "tlg2003.tlg006.perseus-grc1.xml",
]
# TODO it seems like BeautifulSoup only has trouble with about half of these now. Find out which ones are still causing trouble and remove the rest from the list.
FILES_CAUSING_PARSING_ERRORS = [
    "tlg2003.tlg013.perseus-grc1.xml",
    "tlg2003.tlg017.perseus-grc1.xml ",
    "tlg2040.tlg002.perseus-grc1.xml",
    "tlg2040.tlg004.perseus-grc1.xml",
    "tlg0648.tlg001.perseus-grc1.xml",
    "tlg2018.tlg002.perseus-grc1.xml",
    "tlg0363.tlg007.perseus-grc1.xml",
    "tlg0058.tlg001.perseus-grc1.xml",
    "tlg2003.tlg017.perseus-grc1.xml",
    "tlg0099.tlg001.perseus-grc1.xml",
    "tlg0556.tlg001.perseus-grc1.xml",
    "tlg0019.tlg007.perseus-grc1.xml",
    "tlg0019.tlg007.perseus-grc1.xml",
    "tlg0284.tlg029.perseus-grc1.xml",
    "tlg0284.tlg026.perseus-grc1.xml",
    "tlg0284.tlg046.perseus-grc1.xml",
    "tlg0284.tlg045.perseus-grc1.xml",
    "tlg0284.tlg048.perseus-grc1.xml",
    "tlg0284.tlg054.perseus-grc1.xml",
    "tlg0284.tlg009.perseus-grc1.xml",
    "tlg0284.tlg004.perseus-grc1.xml",
    "tlg0284.tlg035.perseus-grc1.xml",
    "tlg0284.tlg022.perseus-grc1.xml",
    "tlg0641.tlg001.perseus-grc1.xml",
]
PERSEUS_FILES_TO_EXCLUDE = (
    BACHCHYLIDES_ODES
    + BETA_CODE_FILES
    + FILES_CAUSING_PARSING_ERRORS
    + TEXTS_WITH_SIGNIFICANT_OCR_ERRORS
)
CHARS_TO_REMOVE = """{}΄|Ϛݲ§ϛ♃5ᾱᾅὝᾂ̆ᾦ#Ἦ*ᾆ⟩ὋἎὒὮ′̣ϝὯἾ͵ῂüὬ⌋⌈‚•ä+̀ö&–ͅᾕë1͂ῲᾡἇἛἋGϠ¶%ῢ^ἊἯæᾇ\\2ᾁῡἚ̋/⌉Ὢß!⌊Ἣ=ῗóΌ`3ï⌞⌟ᾲΆ65ϡ̈4∗Ὣq═òΈϞ○à7áΊᾒ‖~אϟΪϥ›\u200b⁄‹íé⋖ὊÍ9Ἲ̄8{ῧ}Üᾟᾍᾨ―ΉŕὟ⩹✶0Ώᾯᾥᾌ\x8d⟦⟧\x9a¦ᾬἻ£a⋇ῐ¬ÓbῚŒἿἏÉῌᾃ\x98°ΎῈÁ⨆�
ç↑ạὛ⏔⏑̅✝ú\x9dᾺụᾢᾓᾘῼùÒSῠϙ─\x90לṕᾣô\x9cῸᾜḿ$⦵⊏ī\x9eֹ\x8eÌćÆ\x8fǁ⊻@ū÷Ҁ∾ῺìῪ\x8c\x81ᾮᾈèÿœῩῊ\x88⊤З♀⊙\xadÄÖᾞߤ⁑⸨\x8aḍ⫯∼źẂ⋆★Ῑᾩ‵ᾎý√⏝⏓⏕ṃ×ȳហḾti¿⥽⥼⊣⊔ӄẉ͎\u070eҏďĎ̠◻ᾰ\ue1c2rƑ̧\x7fេឲិតា
ỳӕῘLᾙΫẃ☾☿♂♄⊢⋃Ā±TMĹ€║̇čō"""

CHARS_TO_REPLACE = {
    "∠": "Δ",
    "△": "Δ",
    "\ufeff": " ",  # ZERO WIDTH NO-BREAK SPACE (U+FEFF)
    "ῑ": "ϊ",
    "✝": "†",
    # regularize angled brackets
    "<": "⟨",
    ">": "⟩",
    "〈": "⟨",
    "〉": "⟩",
    # regularize quotation marks
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "„": '"',
    "‟": '"',
    "‹": "'",
    "›": "'",
    "‘": "'",
    "": "'",
}


def get_files(directory, regex, files_to_exclude):
    """Finds files matching the regex in the specified directory except for those in the exclusion list. Returns a list of file paths."""
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if re.search(regex, f) and f not in files_to_exclude:
                files.append(dirpath + "/" + f)
    print(f"Number of texts found: {len(list(files))}")
    return files


def parse_xml(fp):
    """Parses a Perseus XML file. The approch here is very rough. Ideally the nodes should be traversed and those containing a significan amount of non-greek characters removed (see: https://github.com/ThomasK81/TEItoCEX)"""
    soup = BeautifulSoup(fp, "xml")
    # remote 'note' tags
    for note_tag in soup.find_all("note"):
        note_tag.decompose()
    assert len(list(soup.find_all("note"))) == 0
    # get all remaining text
    raw_text = soup.find("text").get_text()
    return raw_text


def parse_perseus_xml(files):
    """Parses the given XML files. Returns the parsed files as well as a list of files which couldn't be parsed at all."""
    raw_texts = []
    failed_to_parse = []
    print("Parsing: ", end="")
    for i, f in enumerate(files):
        raw_text = ""
        with open(f) as fp:
            raw_text = parse_xml(fp)
        if not raw_text:
            failed_to_parse.append(f)
            print("x", end="", flush=True)
        else:
            raw_texts.append(raw_text)
            print(".", end="", flush=True)
    print(f"\nNumber of texts parsed: {len(raw_texts)}")
    print(f"Number of texts which could not be parsed: {len(failed_to_parse)}")
    return raw_texts, failed_to_parse


def remove_braces(text):
    """Removes various braces and brackets from the text."""
    # remove square and curly braces and their contents (if less than 40 chars)
    text = re.sub(r"\{[^}]{,40}\}", "", text)
    # remove text between ⟦ and ⟧ as the pairs were manually checked, there is no length restriction
    text = re.sub(r"⟦[^⟧]*⟧", "", text)
    return text


def remove_unwanted_chars(token, chars_to_remove):
    """Removes all characters is the list from the token."""
    output = filter(lambda c: c not in chars_to_remove, token)
    return "".join(output)


def replace_chars(token, chars_to_replace):
    """Replaces all characters in the dict with their specified replacement."""
    for c in token:
        if c in chars_to_replace:
            token = re.sub(c, chars_to_replace[c], token)
    return token


def clean_tokens(tokens, chars_to_remove, chars_to_replace):
    """Cleans a list of tokens."""
    cleaned_tokens = []
    for t in tokens:
        if t:
            # remove words in which latin characters appear
            if not re.search(r"\w+", t, re.ASCII):
                # remove tokens containing digits
                if not re.search(r"\d+", t):
                    # normalize
                    t = cltk_normalize(t)
                    t = t.strip("\t\r\n")
                    # remove unwanted chars
                    t = remove_unwanted_chars(t, chars_to_remove)
                    t = replace_chars(t, chars_to_replace)
                    # remove any inter-word hypens or en-dashes
                    t = re.sub(r"([^\s])(-|–)", r"\1", t)
                    # convert some other forms of whitespace to normal spaces
                    t = re.sub(r"\s+", " ", t)
                    # remove repeated whitespace
                    t = re.sub(r"\s{2,}", " ", t)
                    cleaned_tokens.append(t)
    return cleaned_tokens


def clean_texts(raw_texts, chars_to_remove, chars_to_replace):
    """Cleans a list of texts."""
    data = []
    print("Cleaning: ", end="")
    for i, raw_text in enumerate(raw_texts):
        text = ""
        raw_text = remove_braces(raw_text)
        raw_text = raw_text.splitlines()
        for line in raw_text:
            if line != "\n":
                tokens = line.split(" ")
                cleaned_tokens = clean_tokens(tokens, chars_to_remove, chars_to_replace)
                tokens = " ".join([t for t in cleaned_tokens])
                text += tokens.strip("\n") + " "
        # remove empty parentheses
        text = re.sub(r"\(\s*\)", "", text)
        # remove empty angled brackets
        text = re.sub(r"⟨\s*⟩", "", text)
        # final whitespace pass
        text = re.sub(r"\s{2,}", " ", text)
        data.append(text.strip(" "))
        print(".", end="", flush=True)
    print(f"\nNumber of texts cleaned: {len(data)}")
    return data


def get_f1kg_texts(files):
    """Gets the specified F1KG text files (which do not need parsing, unlike the Perseus files."""
    texts = []
    for i, f in enumerate(files):
        with open(f, "r") as fp:
            texts.append(fp.read())
    print(f"Number of texts read: {len(texts)}")
    return texts


def clean_data():
    """Cleans the Perseus and F1KG data to produce the dataset."""
    print("Cleaning Perseus texts")
    perseus_dir = "canonical-greekLit/data"
    perseus_regex = re.compile("grc[0-9]*\.xml$")
    perseus_files = get_files(perseus_dir, perseus_regex, PERSEUS_FILES_TO_EXCLUDE)
    perseus_raw_texts, failed_to_parse = parse_perseus_xml(perseus_files)
    if failed_to_parse:
        write_to_file("perseus_parsing_failures.txt", failed_to_parse)
    perseus_data = clean_texts(perseus_raw_texts, CHARS_TO_REMOVE, CHARS_TO_REPLACE)

    print("Cleaning F1KG texts")
    f1kg_dir = glob.glob("OpenGreekAndLatin*/text")[0]
    f1kg_regex = re.compile("grc[0-9]*\.txt$")
    f1kg_files = get_files(f1kg_dir, f1kg_regex, [])
    f1kg_raw_texts = get_f1kg_texts(f1kg_files)
    f1kg_data = clean_texts(f1kg_raw_texts, CHARS_TO_REMOVE, CHARS_TO_REPLACE)

    all_texts = perseus_data + f1kg_data
    write_to_file("Ancient_Greek_ML.txt", all_texts)


if __name__ == "__main__":
    clean_data()
