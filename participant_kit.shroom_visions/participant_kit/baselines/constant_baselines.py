import pandas as pd
import pathlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', help='path to downloaded SHROOM-Visions data')
parser.add_argument('output_dir', help='path where constant predictions will be saved')
args = parser.parse_args()



possible_labels = ['invention', 'mischaracterization', 'OCR', 'miscounting', 'other']

for file in pathlib.Path(args.input_dir).glob('shroom-vision.test.*.jsonl'):
    df = pd.read_json(file, lines=True)
    lang = file.name.split('.')[2]
    df['text_len'] = df.response.str.len()
    df['labels'] = [[] for _ in range(len(df))]
    df[['id', 'labels']].to_json(f'{args.output_dir}/{lang}.mark_none.jsonl', lines=True, orient='records')
    for label in possible_labels:
        df['labels'] = df.text_len.apply(
            lambda length: [{'start': 0, 'end': length, 'label': label, 'prob': 1.0}]
        )
        df[['id', 'labels']].to_json(f'{args.output_dir}/{lang}.mark_{label}.jsonl', lines=True, orient='records')

