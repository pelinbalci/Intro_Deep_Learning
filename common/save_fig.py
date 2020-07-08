import os
from folders import folder


def save_fig(folder_name, fig, name):
    file_name = name + '.png'
    path = os.path.join(folder['main_folder'], folder_name, file_name)
    fig.savefig(path)