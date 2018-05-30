from gensim import corpora
from gensim.corpora import Dictionary
import re
from requests import get
from collections import defaultdict
from typing import List, Dict
import argparse


def remove_stopwords(documents: List[str]) -> List[List[str]]:
    stop_list = set('for a of the and to in'.split())
    return [[word for word in document.lower().split() if word not in stop_list] for document in documents]


def remove_hapax_legomena(texts: List[List[str]]):
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    return [[token for token in text if frequency[token] > 1] for text in texts]


def build_dictionary(texts: List[List[str]]) -> Dictionary:
    dictionary = corpora.Dictionary(texts)
    return dictionary


def get_sense_id(url: str) -> Dict[str, str]:
    r = get(f"{url}/senses/random/")
    j = r.json()
    return {"id": j["xml_id"], "headword": j["headword"]}


def get_sense_examples(url: str, sense_id: str) -> List[str]:
    r = get(f"{url}/senses/{sense_id}/remaining_examples/")
    j = r.json()
    return [ex["lyric"] for ex in j["examples"]]


def slugify(value):
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value


def main(url: str):
    from pprint import pprint
    sense = get_sense_id(url)
    headword = sense["headword"]
    print(headword)
    fn = slugify(headword)
    docs = get_sense_examples(url, sense["id"])
    txts = remove_stopwords(docs)
    txts_minus_hapax = remove_hapax_legomena(txts)
    dictionary = build_dictionary(txts_minus_hapax)
    dictionary.save(f"resources/{fn}")
    pprint(dictionary.token2id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    args = parser.parse_args()
    main(args.url)
