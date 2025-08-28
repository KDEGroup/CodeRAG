import os
import json

from coderag.static_analysis.data_flow.graph import tGraph
from coderag.static_analysis.data_flow.extract_dataflow import PythonParser
from coderag.static_analysis.data_flow.node_prompt import projectSearcher
from coderag.static_analysis.data_flow.utils import MAX_HOP, ONLY_DEF, ENABLE_DOCSTRING, LAST_K_LINES
from typing import Callable


class Generator(object):
    """
    Generator class for static analysis and data flow processing.
    This class provides methods to parse Python source code, analyze data flow,
    and generate prompts for code understanding tasks. It integrates with a 
    Python parser, project searcher, and tokenizer to handle project-specific 
    information and cross-file imports.
    Attributes:
        parser (PythonParser): An instance of the PythonParser for parsing source code.
        proj_dir (str): Absolute path to the project directory.
        info_dir (str): Absolute path to the directory containing project information files.
        searcher (projectSearcher): An instance of the projectSearcher for handling project-specific searches.
        tokenizer (ModelTokenizer): An instance of the ModelTokenizer for handling tokenization tasks.
        project (str): The current project being analyzed.
        proj_info (dict): Information about the current project.
    Methods:
        __init__(proj_dir, info_dir, model):
            Initializes the Generator with project and model information.
        _set_project(project):
            Sets the current project and loads its associated information.
        set_pyfile(project, fpath):
            Sets the Python file for analysis and updates project information.
        _get_module_name(fpath):
            Converts a file path to its corresponding Python module name.
        get_suffix(fpath):
            Retrieves the suffix (e.g., comments) associated with a file path.
        sort_by_lineno(src_list, reverse=True):
            Sorts a list of items by their line numbers.
        get_cross_file_nodes(fpath, imported_info):
            Identifies cross-file import nodes based on imported information.
        get_prompt(node_list):
            Generates a prompt based on a list of nodes.
        retrieve_prompt(project, fpath, source_code):
            Retrieves a prompt for a given project and Python file by analyzing
            the source code and cross-file imports.
    """
    def __init__(self, proj_dir, info_dir):
        self.parser = PythonParser()
        self.proj_dir = os.path.abspath(proj_dir)
        self.info_dir = os.path.abspath(info_dir)

        self.searcher = projectSearcher()

        self.project = None
        self.proj_info = None
    

    def _set_project(self, project):
        if project == self.project:
            return

        info_file = os.path.join(self.info_dir, f'{project}.json')
        if not os.path.isfile(info_file):
            print(f'Unknown package {project} in {self.info_dir}')
            return
        
        self.project = project
        with open(info_file, 'r') as f:
            self.proj_info = json.load(f)
    

    def set_pyfile(self, project, fpath):
        self._set_project(project)
        
        # remove current file
        if fpath in self.proj_info:
            proj_info = {k:v for k,v in self.proj_info.items() if k != fpath}
        else:
            proj_info = self.proj_info
        
        dir_path = os.path.join(self.proj_dir, project)
        if os.path.isdir(dir_path):
            content = list(os.listdir(dir_path))
            if len(content) == 1:
                dir_path = os.path.join(dir_path, content[0])
        
        self.searcher.set_proj(dir_path, proj_info)
    

    def _get_module_name(self, fpath):
        if fpath.endswith('.py'):
            fpath = fpath[:-3]
            if fpath.endswith('__init__'):
                fpath = fpath[:-8]

        fpath = fpath.rstrip(os.sep)
        return fpath[len(self.searcher.proj_dir):].replace(os.sep, '.')


    def get_suffix(self, fpath):
        return self.searcher.get_path_comment(fpath)
    

    def sort_by_lineno(self, src_list, reverse=True):
        '''
        src_list: [(anything, ..., lineno)]
        Return: sorted list without lineno
        '''
        sorted_list = sorted(src_list, key=lambda x:x[-1], reverse=reverse)
        return [x[:-1] for x in sorted_list]


    def get_cross_file_nodes(self, fpath, imported_info):
        node_list = []
        for item in imported_info:
            find_info = self.searcher.is_local_import(fpath, item)
            if find_info is not None:
                if find_info[1] is None:
                    find_info = (find_info[0], '')
                else:
                    find_info = tuple(find_info)
                
                if find_info not in node_list:
                    node_list.append(find_info)
        
        return node_list
    

    def get_prompt(self, node_list):
        return self.searcher.get_prompt(node_list, MAX_HOP, ONLY_DEF, ENABLE_DOCSTRING)
    
    def get_prompt_list(self, node_list) -> list[str]:
        return self.searcher.get_prompt_list(node_list, MAX_HOP, ONLY_DEF, ENABLE_DOCSTRING)


    # retrieve until truncated
    def retrieve_prompt_list(self, project, fpath, source_code, calc_truncated: Callable[[list[str]], bool]) -> list[str]:
        self.set_pyfile(project, fpath)

        fpath = self._get_module_name(fpath)

        self.parser.parse(source_code)

        limit_assign = True
        graph = tGraph(self.parser.DFG)

        # check cross-file imports
        cross_import_nodes = set()
        for k, v in graph.node_dict.items():
            if v.node_type == 'import' and self.searcher.is_local_import(fpath, (v.module, v.name)) is not None:
                cross_import_nodes.add(k)

        # Part1: imported information from last k lines
        variable_nodes = graph.get_last_k_lines(LAST_K_LINES)
        related_nodes = graph.get_related_nodes(variable_nodes, reverse=True, limit_assign=limit_assign)
        proj_nodes = set(related_nodes) & cross_import_nodes
        
        # subgraph
        related_nodes = graph.get_related_nodes(variable_nodes, reverse=True, end_nodes=proj_nodes, limit_assign=limit_assign)
        # create subgraph
        subgraph = graph.get_assign_subgraph(related_nodes, proj_nodes)
        # all nodes with module info in subgraph
        imported_dict = {}
        for k, v in subgraph.module_info.items():
            for item in v:
                # (module, name)
                info = tuple(item[:2])
                # pos: smaller is better (the lineno of import statements)
                pos = (0, item[2])
                if info not in imported_dict:
                    imported_dict[info] = pos
                else:
                    imported_dict[info] = min(imported_dict[info], pos)

        # Part2: other import nodes
        other_proj_nodes = cross_import_nodes - proj_nodes
        related_nodes = graph.get_related_nodes(other_proj_nodes, reverse=False, end_nodes=None, limit_assign=True)
        # create subgraph
        subgraph = graph.get_assign_subgraph(related_nodes, other_proj_nodes)
        # all nodes with module info in subgraph
        other_imported_dict = {}
        for k, v in subgraph.module_info.items():
            for item in v:
                # (module, name)
                info = tuple(item[:2])
                # pos: smaller is better (the lineno of import statements)
                pos = (1, item[2])
                if info not in other_imported_dict:
                    other_imported_dict[info] = pos
                else:
                    other_imported_dict[info] = min(other_imported_dict[info], pos)
        

        # prompt from Part 1
        imported_info = self.sort_by_lineno([(k[0], k[1], v) for k, v in imported_dict.items()])
        node_list = self.get_cross_file_nodes(fpath, imported_info)
        prompt_list = self.get_prompt_list(node_list)

        # other imported info from Part 2
        sorted_others = sorted(other_imported_dict, key=lambda x: other_imported_dict[x])
        for item in sorted_others:
            if item not in imported_dict:
                imported_dict[item] = other_imported_dict[item]
            else:
                imported_dict[item] = min(imported_dict[item], other_imported_dict[item])

            imported_info = self.sort_by_lineno([(k[0], k[1], v) for k, v in imported_dict.items()])
            node_list = self.get_cross_file_nodes(fpath, imported_info)
            new_prompt_list = self.get_prompt_list(node_list)

            # Check if the new prompt exceeds the maximum length
            if len(prompt_list) > 0 and calc_truncated(new_prompt_list):
                break

            prompt_list= new_prompt_list

        return prompt_list