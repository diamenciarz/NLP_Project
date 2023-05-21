import numpy as np
import pandas as pd
import json


def __parse_json_record(record):
    record_dict = [json.loads(record)]
    record_df = pd.DataFrame(record_dict)
    record_df.drop(["id", "authors", "submitter", "license", "journal-ref", "update_date", "comments", "report-no", "doi"], axis=1, inplace=True)
    return record_df

def read_n_documents(start_reading_at = 0, document_count = 5, path = 'arxiv-metadata-oai-snapshot.json'):
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