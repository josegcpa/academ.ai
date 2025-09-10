import sqlite3
from multiprocessing import Pool
from datetime import datetime
from datetime import timedelta
import logging

from . import retrievers
from . import db

logger = logging.getLogger("academic_search")

schemas = {
    "papers": db.Paper.model_json_schema(),
    "authors": db.Author.model_json_schema(),
    "author_affiliations": db.AuthorAffiliation.model_json_schema(),
}


def retrieve_biorxiv_medrxiv_mp_wrapper(kwargs):
    return retrievers.retrive_biorxiv_medrxiv(**kwargs), kwargs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, choices=["biorxiv", "medrxiv"], default="biorxiv"
    )
    parser.add_argument("--date_start", type=str, required=True)
    parser.add_argument("--date_end", type=str, required=True)
    parser.add_argument("--output_db", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=0)
    args = parser.parse_args()

    date_start = datetime.strptime(args.date_start, "%Y-%m-%d")
    date_end = datetime.strptime(args.date_end, "%Y-%m-%d")

    logger.info(
        f"Retrieving {args.source} from {date_start.strftime('%Y-%m-%d')} to {date_end.strftime('%Y-%m-%d')}"
    )

    db.create_db(args.output_db)

    exclude_dates = []
    exclude_dois = []
    with sqlite3.connect(args.output_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT date_source FROM papers WHERE source = ?",
            (args.source,),
        )
        dates = [row[0] for row in cursor.fetchall()]
        if len(dates) > 0:
            exclude_dates.extend(
                [
                    datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
                    for date in dates
                ]
            )
        cursor.execute(
            "SELECT DISTINCT date FROM dates_with_no_papers WHERE source = ?",
            (args.source,),
        )
        dates = [
            datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
            for row in cursor.fetchall()
        ]
        exclude_dates.extend(dates)
        cursor.execute(
            "SELECT doi FROM papers WHERE source = ?",
            (args.source,),
        )
        dois = [row[0] for row in cursor.fetchall()]
        exclude_dois.extend(dois)
    exclude_dates = list(set(exclude_dates))

    days = [
        date_start + timedelta(days=i)
        for i in range((date_end - date_start).days + 1)
    ]
    logger.info(f"Total days: {len(days)}")
    days = [d for d in days if d.strftime("%Y-%m-%d") not in exclude_dates]
    logger.info(f"Filtered days: {len(days)}")

    all_date_kwargs = [
        {"source": args.source, "date_start": d, "date_end": d} for d in days
    ]
    if args.n_workers > 1:
        pool = Pool(args.n_workers)
        map_fn = pool.imap
    else:
        map_fn = map

    for day_idx, (output, kwargs) in enumerate(
        map_fn(retrieve_biorxiv_medrxiv_mp_wrapper, all_date_kwargs)
    ):
        current_date = kwargs["date_start"]
        with sqlite3.connect(args.output_db) as conn:
            five_days_ago = datetime.now() - timedelta(days=5)
            if len(output) == 0 and current_date < five_days_ago:
                db.insert_into_table(
                    conn,
                    "dates_with_no_papers",
                    {"date": current_date, "source": args.source},
                )
            logger.info(f"Inserting {len(output)} papers")
            for data_entry in output:
                paper = data_entry["article"]
                authors = data_entry["authors"]
                author_affiliations = data_entry["author_affiliations"]
                doi = paper["doi"]
                if doi in exclude_dois:
                    continue
                author_ids_affiliations = []
                try:
                    paper_id = db.insert_into_table(
                        conn,
                        "papers",
                        {
                            k: paper[k]
                            for k in schemas["papers"]["properties"].keys()
                        },
                    )
                except Exception as e:
                    import pprint

                    logger.error(f"Error inserting paper: {paper['doi']}")
                    pprint.pprint(paper)
                    raise e
                for author_idx, author in enumerate(authors):
                    data_dict = {
                        "first_name": author["first_name"],
                        "last_name": author["last_name"],
                        "paper_id": paper_id,
                        "author_idx": author_idx,
                        "corresponding": author["corresponding"],
                    }
                    author_id = db.insert_into_table(conn, "authors", data_dict)
                    if author["corresponding"]:
                        author_ids_affiliations.append(author_id)
                for author_affiliation, author_id in zip(
                    author_affiliations, author_ids_affiliations
                ):
                    data_dict = {
                        "author_id": author_id,
                        "institution": author_affiliation["institution"],
                        "date_source": author_affiliation["date_source"],
                    }
                    db.insert_into_table(conn, "author_affiliations", data_dict)
                conn.commit()
                exclude_dois.append(doi)
        logger.info(f"Inserted {len(output)} papers")
        logger.info(f"Finished {day_idx + 1} out of {len(days)}")

    if args.n_workers > 1:
        pool.close()

if __name__ == "__main__":
    main()
