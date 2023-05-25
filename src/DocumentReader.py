import numpy as np
import pandas as pd
import json

def __shorten_version(versions):
    return int(versions[-1]["created"].split()[3])

def __parse_json_record(record):
    record_dict = [json.loads(record)]
    record_df = pd.DataFrame(record_dict)
    version = record_df["versions"].apply(lambda x: __shorten_version(x))
    record_df.insert(8,"version", version)
    record_df.drop(["id", "authors", "submitter", "license", "journal-ref", "update_date", "comments", "report-no", "doi", "versions", category_names], axis=1, inplace=True)

    return record_df

def read_n_documents(start_reading_at = 0, document_count = 5, path = 'data//arxiv-metadata-oai-snapshot.json'):
    """
    Reads n number of documents from the JSON file. You can make it start reading at any position
    """
    with open(path, 'r') as file:
        records = pd.DataFrame()
        i = 0
        for line in file:
            if i < start_reading_at:
                i += 1
                continue
            parsed_line = __parse_json_record(line)
            if i == start_reading_at:
                records = parsed_line
                records.index = [i]
                i += 1
                continue
            records.loc[i] = parsed_line.values[0]
            i += 1
            if i >= document_count:
                break
    return records