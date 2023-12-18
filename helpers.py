import datetime
import json
import logging
import os
import time

import valohai

logger = logging.getLogger(__name__)


def save_valohai_metadata(model, output_dir):
    model.save_pretrained(output_dir)
    project_name, exec_id = get_run_identification()
    metadata = {
        'valohai.dataset-versions': [
            {
                'uri': f'dataset://llm-models/{project_name}_{exec_id}',
                'targeting_aliases': [f'dev-{datetime.date.today()}-model'],
                'valohai.tags': ['dev', 'llm'],
            },
        ],
    }
    for file in os.listdir(output_dir):
        md_path = os.path.join(output_dir, f'{file}.metadata.json')
        metadata_path = valohai.outputs().path(md_path)
        with open(metadata_path, 'w') as outfile:
            json.dump(metadata, outfile)


def get_run_identification():
    try:
        with open('/valohai/config/execution.json') as f:
            exec_details = json.load(f)
        project_name = exec_details['valohai.project-name'].split('/')[1]
        exec_id = exec_details['valohai.execution-id']
    except FileNotFoundError:
        project_name = 'test'
        exec_id = str(int(time.time()))
    return project_name, exec_id
