import string
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

RELEVANT_CHAR = string.ascii_letters + string.digits + string.punctuation + " "
REPLACE_DICT = {
    "ü": "u"
}

ARTICLE_KEYS = [
    "title", 
    "abstract", 
    "journal", 
    "funding", 
    "license",
    "category", 
    "date_source",
    "doi", 
    "xml", 
    "xml_format",
    "source", 
    "date_added", 
    "preprint", 
    "published_doi"
]

def process_string(str_input: str)->str:
    # transform non-ascii characters to ascii
    for k, v in REPLACE_DICT.items():
        str_input = str_input.replace(k, v)
    str_input = str_input.encode('ascii', 'ignore').decode('ascii')
    str_input = ''.join(c for c in str_input.strip() if c in RELEVANT_CHAR)
    return str_input

def retrive_biorxiv_medrxiv(source: str, date_start: datetime, date_end: datetime, stop_after_first: bool = False):
    assert source in ["biorxiv", "medrxiv"]
    URL = f"https://api.biorxiv.org/details/{source}/{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}"
    logger.info(f"Retrieving {URL}")
    response = requests.get(URL)
    response.raise_for_status()
    json_response = response.json()
    message = json_response["messages"][0]
    if message["status"] == "no posts found":
        logger.info("No posts found")
        return []
    if message["status"] != "ok":
        raise Exception("Error: " + message["status"])
    total_n_papers = int(message["total"])
    collection = json_response["collection"]
    logger.info(f"Total number: {total_n_papers} papers")
    logger.info(f"Collection has {len(collection)} papers")
    output_collection = []
    cursor = 0
    while len(output_collection) < total_n_papers:
        if cursor > 0:
            logger.info(f"Retrieving additional papers starting from {cursor}")
            URL = f"https://api.biorxiv.org/details/{source}/{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}/{cursor}"
            response = requests.get(URL)
            response.raise_for_status()
            json_response = response.json()
            message = json_response["messages"][0]
            if message["status"] != "ok":
                raise Exception("Error: " + message["status"])
            collection = json_response["collection"]
            logger.info(f"Collection has {len(collection)} papers")
        for idx, article in enumerate(collection):
            authors = []
            author_affiliations = []
            collection[idx]["date_source"] = datetime.strptime(
                article["date"], "%Y-%m-%d").date()
            collection[idx]["authors_clean"] = []
            for names in article["authors"].split(";"):
                if len(names) == 0:
                    continue
                if "," in names:
                    names = names.split(",")
                else:
                    names = names.split()
                    names = [names[-1], " ".join(names[:-1])]
                if len(names) == 1:
                    continue
                authors.append({
                    "last_name": process_string(names[0]), 
                    "first_name": process_string(names[1]),
                })
                corresponding = (
                    authors[-1]["last_name"] 
                    in process_string(collection[idx]["author_corresponding"])
                )
                authors[-1]["corresponding"] = corresponding
                if corresponding:
                    author_affiliations.append({
                        "institution": process_string(collection[idx]["author_corresponding_institution"]),
                        "date_source": collection[idx]["date_source"]
                    })
            collection[idx]["preprint"] = True
            collection[idx]["published_doi"] = collection[idx]["published"]
            collection[idx]["xml"] = collection[idx]["jatsxml"]
            collection[idx]["xml_format"] = "jats"
            collection[idx]["source"] = source
            collection[idx]["date_added"] = datetime.now()
            collection[idx]["journal"] = collection[idx]["server"]
            if "funding" in collection[idx]:
                collection[idx]["funding"] = str(collection[idx]["funding"])
            else:
                collection[idx]["funding"] = ""
            output_collection.append({
                "article": {
                    k: collection[idx][k] 
                    for k in ARTICLE_KEYS
                },
                "authors": authors,
                "author_affiliations": author_affiliations,
            })
        cursor += len(collection)
        logger.info(f"Retrieved {len(output_collection)} papers")
        if stop_after_first:
            return output_collection
    return output_collection

if __name__ == "__main__":
    from pprint import pprint
    pprint(
        retrive_biorxiv_medrxiv(
            "biorxiv", datetime(2022, 1, 1), datetime(2022, 1, 31)
        )
    )
