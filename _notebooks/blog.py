import nbconvert
from traitlets.config import Config
from nbconvert.preprocessors import Preprocessor

from IPython.display import display, Javascript

NOTEBOOK_FULL_PATH = '...'


def set_global_notebook_name():
    display(
        Javascript('''
        var nb = IPython.notebook
        var kernel = IPython.notebook.kernel
        var command = "NOTEBOOK_FULL_PATH = '" + nb.base_url + nb.notebook_path + "'"
        kernel.execute(command)
    '''))


def save_notebook():
    display(Javascript('''
        IPython.notebook.save_notebook();
    '''))


def post_this_notebook(notebook_path, draft=False):
    save_notebook()

    full_notebook_name = notebook_path.split('/')[-1]
    notebook_name = full_notebook_name.split('.')[0]

    folder = '_drafts' if draft else '_posts'
    output_name = f'../{folder}/{notebook_name}.md'

    convert_notebook(full_notebook_name, output_name)


def convert_notebook(notebook_path, notebook_output):
    class FilterCells(Preprocessor):
        def filter_cell(self, cell):
            cell_text = cell['source'].strip().lower()
            return '# exclude cell' not in cell_text

        def map_cell(self, cell):
            cell_text = cell['source'].strip().lower()
            is_hidden = cell['metadata'].get('hide_input', False)

            if is_hidden or '# exclude input' in cell_text:
                cell = dict(
                    cell_type='raw',
                    metadata=cell.metadata,
                    source='\n'.join(
                        [o['data'] for o in cell['outputs'] if 'text' in o]),
                )

            return cell

        def preprocess(self, nb, resources):
            global CELLS
            CELLS = nb.cells
            nb.cells = [
                self.map_cell(c) for c in nb.cells if self.filter_cell(c)
            ]

            return nb, resources

    c = Config()
    c.MarkdownExporter.preprocessors = [FilterCells]
    exporter = nbconvert.MarkdownExporter(config=c)
    body, _resources = exporter.from_filename(notebook_path)

    with open(notebook_output, 'w+') as f:
        f.write(body)


def post():
    import os
    import sys
    import json

    input_file = sys.argv[1]

    base_name = os.path.basename(input_file)
    default_out_file_nama = f'{os.path.splitext(base_name)[0]}.md'
    out_file_name = sys.argv[2] if len(sys.argv) > 2 else default_out_file_nama

    print(f'>> Converting {input_file} to blog post')
    print(f'>> Out file {out_file_name}')
    with open(input_file, 'r') as f:
        notebook_json = json.load(f)
        meta_cell = notebook_json['cells'][0]['source']
        IS_DRAFT = any('draft: true' in c.lower() for c in meta_cell)

        post_text = ''
        for c in notebook_json['cells']:
            cell_text = ''.join(c['source'])

            if '# hide' not in cell_text:
                post_text += cell_text
                post_text += '\n'

        out_path = '_drafts' if IS_DRAFT else '_posts'
        out_path += f'/{out_file_name}'

        with open(out_path, 'w+') as f:
            f.write(post_text)


if __name__ == '__main__':
    # Example usage:
    # py _notebooks/blog.py _notebooks/2020-27-07-soft-addressable-computation.ipynb
    post()
