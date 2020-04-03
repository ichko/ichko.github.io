import nbconvert
from traitlets.config import Config
from nbconvert.preprocessors import Preprocessor

from IPython.display import display, Javascript


def set_global_notebook_name():
    display(Javascript('''
        var nb = IPython.notebook
        var kernel = IPython.notebook.kernel
        var command = "NOTEBOOK_FULL_PATH = '" + nb.base_url + nb.notebook_path + "'"
        kernel.execute(command)
    '''))


def post_this_notebook(notebook_path, draft=False):
    full_notebook_name = notebook_path.split('/')[-1]
    notebook_name = full_notebook_name.split('.')[0]

    folder = '_drafts' if draft else '_posts'
    output_name = f'../{folder}/{notebook_name}.md'

    convert_notebook(full_notebook_name, output_name)


def convert_notebook(notebook_path, notebook_output):
    class FilterCells(Preprocessor):
        def filter_cell(self, cell):
            cell_text = cell['source'].strip().lower()
            return not cell_text.startswith('# exclude cell')

        def map_cell(self, cell):
            cell_text = cell['source'].strip().lower()
            if cell_text.startswith('# exclude code'):
                cell = dict(
                    cell_type='raw',
                    metadata=cell.metadata,
                    source='\n'.join([
                        o['text']
                        for o in cell['outputs'] if 'text' in o
                    ]),
                )

            return cell

        def preprocess(self, nb, resources):
            global CELLS
            CELLS = nb.cells
            nb.cells = [self.map_cell(c)
                        for c in nb.cells if self.filter_cell(c)]

            return nb, resources

    c = Config()
    c.MarkdownExporter.preprocessors = [FilterCells]
    exporter = nbconvert.MarkdownExporter(config=c)
    body, _resources = exporter.from_filename(notebook_path)

    with open(notebook_output, 'w+') as f:
        f.write(body)
