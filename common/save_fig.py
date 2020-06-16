import os
from folders import folder


def save_fig(fig, name):
    file_name = name + '.png'
    path = os.path.join(folder['main_folder'], folder['CNN_images'], file_name)
    fig.savefig(path)