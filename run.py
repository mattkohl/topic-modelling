import argparse
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Optional

from gensim import corpora
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from requests import get


def dictionaries_path(): return "resources/dictionaries/trr.dict"


def corpora_path(fn): return f"resources/corpora/{fn}.mm"


def remove_stopwords(documents: List[str]) -> List[List[str]]:
    stop_list = set('for a of the and to in'.split())
    return [[word for word in document.lower().split() if word not in stop_list] for document in documents]


def remove_hapax_legomena(texts: List[List[str]]) -> List[List[str]]:
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    return [[token for token in text if frequency[token] > 1] for text in texts]


def load_corpus(fn) -> Optional[MmCorpus]:
    try:
        corpus = MmCorpus(corpora_path(fn))
    except Exception as e:
        print(f"No corpus yet: {e}")
        return None
    else:
        return corpus


def build_corpus(fn: str, texts: List[List[str]], dictionary: Dictionary) -> List[List[Tuple[int, int]]]:
    vectors = [dictionary.doc2bow(text) for text in texts]
    MmCorpus.serialize(corpora_path(fn), vectors)
    corpus = MmCorpus(corpora_path(fn))
    return corpus


def load_dictionary() -> Optional[Dictionary]:
    try:
        dictionary = Dictionary.load(dictionaries_path())
    except Exception as e:
        print(f"No dictionary yet: {e}")
        return None
    else:
        return dictionary


def build_dictionary(texts: List[List[str]]) -> Dictionary:
    dictionary = corpora.Dictionary(texts)
    dictionary.save(dictionaries_path())
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
    document = get_random_document(url)
    pprint(document)
    dictionary = update_dictionary(dictionary, document["texts"])
    corpus = build_corpus(document["name"], document["texts"], dictionary)
    pprint(list(corpus))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    args = parser.parse_args()
    main(args.url)
