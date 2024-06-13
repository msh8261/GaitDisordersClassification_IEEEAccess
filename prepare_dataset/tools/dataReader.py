# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import glob
import os

from prepare_dataset.tools.decorators import (countcall, dataclass, logger,
                                              timeit)


@logger
class DataReader(object):
    def __init__(self, base_dir) -> None:
        self.base_dir = base_dir

    def get_folders_path(self, dir=None):
        if dir is None:
            dir = self.base_dir
        try:
            folders_path = [
                os.path.join(dir, name)
                for name in os.listdir(dir)
                if os.path.isdir(os.path.join(dir, name))
            ]
        except ValueError:
            print("INFO: Please check if the directory of folders exist.")
        return folders_path

    def get_files_path(self, folder_path, format_file=None):
        if format_file is None:
            assert print("please specify the format of file.")
        try:
            files_path = glob.glob(os.path.join(folder_path, f"*.{format_file}"))
        except ValueError:
            print("INFO: Please check if the directory of files exist.")
        return files_path
