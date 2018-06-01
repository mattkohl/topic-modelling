from gensim import corpora
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.test.utils import datapath, get_tmpfile
import re
from requests import get
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Optional
import argparse


def remove_stopwords(documents: List[str]) -> List[List[str]]:
    stop_list = set('for a of the and to in'.split())
    return [[word for word in document.lower().split() if word not in stop_list] for document in documents]


def remove_hapax_legomena(texts: List[List[str]]) -> List[List[str]]:
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    return [[token for token in text if frequency[token] > 1] for text in texts]


def load_corpus() -> Optional[MmCorpus]:
    try:
        corpus = MmCorpus(datapath("resources/corpora/trr.mm"))
    except Exception as e:
        print(f"No corpus yet: {e}")
        return None
    else:
        return corpus


def build_corpus(texts: List[List[str]], dictionary: Dictionary) -> List[List[Tuple[int, int]]]:
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus


def load_dictionary() -> Optional[Dictionary]:
    try:
        dictionary = Dictionary.load(f"resources/dictionaries/trr.dict")
    except Exception as e:
        print(f"No dictionary yet: {e}")
        return None
    else:
        return dictionary


def build_dictionary(texts: List[List[str]]) -> Dictionary:
    dictionary = corpora.Dictionary(texts)
    dictionary.save(f"resources/dictionaries/trr.dict")
    return dictionary


def update_dictionary(dictionary: Optional[Dictionary], texts: List[List[str]]) -> Dictionary:
    if dictionary:
        dictionary.add_documents(texts)
        return dictionary
    else:
        return build_dictionary(texts)


def get_random_sense_id(url: str) -> Dict[str, str]:
    r = get(f"{url}/senses/random/")
    j = r.json()
    return {"id": j["xml_id"], "headword": j["headword"]}


def get_sense_examples(url: str, sense_id: str) -> List[str]:
    r = get(f"{url}/senses/{sense_id}/remaining_examples/")
    j = r.json()
    return [ex["lyric"] for ex in j["examples"]]


def slugify(value: str) -> str:
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value


def get_random_document(url) -> Dict[str, Union[str, List[List[str]]]]:
    sense = get_random_sense_id(url)
    headword = sense["headword"]
    name = f"{slugify(headword)}_{sense['id']}"
    docs = get_sense_examples(url, sense["id"])
    txts = remove_stopwords(docs)
    txts_minus_hapax = remove_hapax_legomena(txts)
    return {"name": name, "texts": txts_minus_hapax}


def main(url: str) -> None:
    from pprint import pprint
    dictionary = load_dictionary()
    corpus = load_corpus()
    document = get_random_document(url)
    pprint(document)
    dictionary = update_dictionary(dictionary, document["texts"])
    pprint(dictionary.token2id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    args = parser.parse_args()
    main(args.url)
