import os
from pathlib import Path
import re
import json

from loguru import logger
from coderag.static_analysis.data_flow.pyfile_parse import PythonParser
from coderag.static_analysis.data_flow.node_prompt import projectSearcher


class projectParser(object):
    """
    Class: projectParser
    This class is responsible for parsing Python projects to extract and retain useful relationships 
    between modules, classes, functions, and variables. It provides methods for setting the project 
    directory, parsing the directory structure, and analyzing intra-file and cross-file relationships.
    Attributes:
        py_parser (PythonParser): An instance of the PythonParser class used for parsing Python files.
        iden_pattern (re.Pattern): A compiled regular expression pattern to identify invalid characters 
            in file or directory names.
        proj_searcher (projectSearcher): An instance of the projectSearcher class used for searching 
            and analyzing project relationships.
        proj_dir (str): The root directory of the project being parsed.
        parse_res (dict): A dictionary containing the parsed results of the project.
    Methods:
        __init__():
            Initializes the projectParser instance with default attributes.
        set_proj_dir(dir_path):
            Sets the root directory of the project to be parsed.
            Args:
                dir_path (str): The path to the project directory.
        retain_project_rels():
            Retains and modifies useful relationships within and across files in the project. 
            Removes invalid or unresolved relationships.
        _get_all_module_path(target_path):
            Recursively retrieves all Python files and directories within the target path.
            Args:
                target_path (str): The root directory to search.
            Returns:
                dict: A dictionary mapping directories to their contained Python files and subdirectories.
        _get_module_name(fpath):
            Converts a file path to a module name relative to the project directory.
            Args:
                fpath (str): The file path to convert.
            Returns:
                str: The module name.
        parse_dir(pkg_dir):
            Parses the given project directory to extract module information and relationships.
            Args:
                pkg_dir (str): The root directory of the project to parse.
            Returns:
                dict: A dictionary containing parsed information about the project structure and relationships.
    """
    def __init__(self):
        self.py_parser = PythonParser()
        self.iden_pattern = re.compile(r'[^\w\-]')

        self.proj_searcher = projectSearcher()

        self.proj_dir = None
        self.parse_res = None
    

    def set_proj_dir(self, dir_path):
        if not dir_path.endswith(os.sep):
            self.proj_dir = dir_path + os.sep
        else:
            self.proj_dir = dir_path


    def retain_project_rels(self):
        '''
        retain the useful relationships
        '''
        for module, file_info in self.parse_res.items():
            for name, info_dict in file_info.items():
                cls = info_dict.get("in_class", None)

                # intra-file relations
                rels = info_dict.get("rels", None)
                if rels is not None:
                    del_index = []
                    for i, item in enumerate(rels):
                        # item: [name, type]
                        find_info = self.proj_searcher.name_in_file(item[0], list(file_info), name, cls)
                        if find_info is None:
                            del_index.append(i)
                        else:
                            # modify
                            info_dict["rels"][i] = [find_info[0], find_info[1], item[1]]
                    
                    # delete
                    for index in reversed(del_index):
                        info_dict["rels"].pop(index)
                    
                    if len(info_dict["rels"]) == 0:
                        info_dict.pop("rels")

                # cross-file relations
                imported_info = info_dict.get("import", None)
                if info_dict["type"] == 'Variable' and imported_info is not None:
                    judge_res = self.proj_searcher.is_local_import(module, imported_info)
                    if judge_res is None:
                        info_dict.pop("import")
                    else:
                        info_dict["import"] = judge_res



    def _get_all_module_path(self, target_path):
        if not os.path.isdir(target_path):
            return {}

        dir_list = [target_path,]
        py_dict = {}
        while len(dir_list) > 0:
            py_dir = dir_list.pop()
            py_dict[py_dir] = set()
            for item in os.listdir(py_dir):
                fpath = os.path.join(py_dir, item)
                if os.path.isdir(fpath):
                    if re.search(self.iden_pattern, item) is None:
                        dir_list.append(fpath)
                        py_dict[py_dir].add(fpath)
                elif os.path.isfile(fpath) and fpath.endswith('.py'):
                    if re.search(self.iden_pattern, item[:-3]) is None:
                        py_dict[py_dir].add(fpath)
        
        return py_dict


    def _get_module_name(self, fpath):
        if fpath.endswith('.py'):
            fpath = fpath[:-3]
            if fpath.endswith('__init__'):
                fpath = fpath[:-8]

        fpath = fpath.rstrip(os.sep)
        return fpath[len(self.proj_dir):].replace(os.sep, '.')


    def parse_dir(self, pkg_dir):
        '''
        Return: {module: {
            name: {
                "type": str,                         # type: "Module", "Class", "Function", "Variable"
                "def": str,
                "docstring": str (optional),
                "body": str (optional),
                "sline": int (optional),
                "in_class": str (optional),
                "in_init": bool (optional),
                "rels": [[name:str, suffix:str, type:str], ],    # type: "Assign", "Hint", "Rhint", "Inherit"
                "import": [module:str, name:str]     # "Import"
            }
            }}
        '''
        self.set_proj_dir(pkg_dir)
        py_dict = self._get_all_module_path(pkg_dir)
        
        # order: dir, __init__.py, .py
        module_dict = {}
        # dir
        for dir_path in py_dict:
            module = self._get_module_name(dir_path)
            if len(module) > 0:
                module_dict[module] = [dir_path,]
        
        # pyfiles
        init_files = set()
        pyfiles = set()
        for py_set in py_dict.values():
            for fpath in py_set:
                if fpath.endswith(os.sep + '__init__.py'):
                    init_files.add(fpath)
                else:
                    pyfiles.add(fpath)
        
        # __init__.py
        for fpath in init_files:
            module = self._get_module_name(fpath)
            if len(module) > 0:
                if module in module_dict:
                    module_dict[module].append(fpath)
                else:
                    module_dict[module] = [fpath,]
        
        # .py
        for fpath in pyfiles:
            module = self._get_module_name(fpath)
            if len(module) > 0:
                if module in module_dict:
                    module_dict[module].append(fpath)
                else:
                    module_dict[module] = [fpath,]
        
        self.parse_res = {}
        for module, path_list in module_dict.items():
            info_dict = {}
            for fpath in path_list:
                if fpath in py_dict:
                    # dir
                    for item in py_dict[fpath]:
                        submodule = self._get_module_name(item)
                        if submodule != module:
                            # exclude __init__.py
                            info_dict[submodule] = {
                                "type": "Module",
                                "import": [submodule, None]
                            }
                else:
                    # pyfiles
                    info_dict.update(self.py_parser.parse(fpath))
                    break
            
            if len(info_dict) > 0:
                self.parse_res[module] = info_dict

        self.proj_searcher.set_proj(pkg_dir, self.parse_res)
        # connect the files
        self.retain_project_rels()

        return self.parse_res


def generate_context_graph(pkg_list: list[str], ds_repo_dir: Path, ds_graph_dir: Path):
    """
    Generate repository-specific context graphs based on the provided package list and directories.

    Args:
        pkg_list (list[str]): List of package names to process.
        ds_repo_dir (Path): Path to the directory containing repositories.
        ds_graph_dir (Path): Path to the directory where context graphs will be saved.
    """
    pkg_set = set(pkg_list)
    logger.info(f'There are {len(pkg_set)} repositories to process.')

    project_parser = projectParser()

    if not ds_graph_dir.is_dir():
        ds_graph_dir.mkdir(parents=True)

    for item in os.listdir(ds_repo_dir):
        if item not in pkg_set:
            continue

        dir_path = ds_repo_dir / item
        if dir_path.is_dir():
            content = list(dir_path.iterdir())
            if len(content) > 1:
                info = project_parser.parse_dir(str(dir_path))
            else:
                # package/package-version/
                dist_path = content[0]
                info = project_parser.parse_dir(str(dist_path))

            with open(ds_graph_dir / f'{item}.json', 'w') as f:
                json.dump(info, f)

    logger.info(f'Generated repo-specific context graphs for {len(list(ds_graph_dir.iterdir()))} repositories.')
