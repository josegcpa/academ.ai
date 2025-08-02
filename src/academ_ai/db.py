import sqlite3
from pydantic import BaseModel
from datetime import datetime

class Paper(BaseModel):
    title: str
    abstract: str
    journal: str
    funding: str
    license: str
    category: str
    date_source: datetime
    doi: str
    xml: str
    xml_format: str
    source: str
    date_added: datetime
    preprint: bool
    published_doi: str
    
class DatesWithNoPapers(BaseModel):
    date: datetime
    source: str

class Author(BaseModel):
    first_name: str
    last_name: str
    paper_id: int
    corresponding: bool

class AuthorAffiliation(BaseModel):
    author_id: int
    institution: str
    date_source: datetime

def data_model_to_sqlite(table_name: str, data_model: BaseModel) -> dict:
    dict_schema = data_model.model_json_schema()
    for property_name, property in dict_schema["properties"].items():
        if property.get("format", None) == "date-time":
            dict_schema["properties"][property_name]["type"] = "DATE"
        elif property["type"] == "string":
            dict_schema["properties"][property_name]["type"] = "TEXT"
        elif property["type"] == "integer":
            dict_schema["properties"][property_name]["type"] = "INTEGER"
        elif property["type"] == "boolean":
            dict_schema["properties"][property_name]["type"] = "BOOL"
    statement = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
    for property_name, property in dict_schema["properties"].items():
        statement.append(f"{property_name} {property['type']}")
    statement = "\n\t" + ", \n\t".join(statement) + "\n"
    statement = f"CREATE TABLE IF NOT EXISTS {table_name} ({statement})"
    return statement


def create_db(db_path: str):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(data_model_to_sqlite("papers", Paper))
        cursor.execute(data_model_to_sqlite("authors", Author))
        cursor.execute(data_model_to_sqlite("author_affiliations", AuthorAffiliation))
        cursor.execute(data_model_to_sqlite("dates_with_no_papers", DatesWithNoPapers))
        conn.commit()

def insert_into_table(conn, table_name: str, data_dict: dict[str]):
    cursor = conn.cursor()
    key_str = ", ".join([f"'{k}'" for k in data_dict.keys()])
    value_list = [v for v in data_dict.values()]
    value_str = ", ".join(["?" for _ in value_list])
    final_str = f"INSERT INTO {table_name} ({key_str}) VALUES ({value_str});"
    cursor.execute(final_str, value_list)
    identifier = cursor.lastrowid
    conn.commit()
    return identifier