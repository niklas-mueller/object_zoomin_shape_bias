import pickle
import os
import csv
import dill
import numpy as np
import pandas as pd
import torch
import yaml

class ResultManager():
    """
    Class to manage any kind of result python object. Will store results in a central location.
    Uses dill as a backend which is more powerful than pickle itself, but will save files in pickle format.
    It allows to load classes that have been modified which is not possible with pickle.
    """

    def __init__(self, root, verbose=True) -> None:
        self.root = root
        self.verbose = verbose

        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

    def _print(self, string:str):
        if self.verbose:
            print(string)

    # def save_model_description(self, model:torch.nn.Module, optimizer='', criterion='', input_channels=0, output_channels=0, overwrite=False):
    #     if isinstance(model, torch.nn.DataParallel):
    #         model = model.module
    #     model_description = {'model_class': type(model), 'model': model, 'optimizer': optimizer, 'criterion': criterion, 
    #                     'input_channels': input_channels, 'output_channels': output_channels, 
    #                     }

    #     self.save_result(result = model_description, filename='model_description.yml', overwrite=overwrite)

    def save_model(self, model, filename:str, path:str=None, overwrite:bool=False):
        if path is None:
            path = self.root
        
        path = os.path.join(path, filename)

        if not overwrite and os.path.exists(os.path.join(path, filename)):
            name, ending = filename.split('.')
            filename = f"{name}_1.{ending}"
            # return
            self._print(f"File exist and will instead be written as {filename}")

        torch.save(model.state_dict(), path)

        self._print(f"Model successfully saved to {path}!")

    def save_result(self, result, filename:str, path:str=None, overwrite:bool=False):

        if path is None:
            path = self.root
        
        if not overwrite and os.path.exists(os.path.join(path, filename)):
            s = filename.split('.')
            name = "".join(s[:-1])
            ending = s[-1]
            filename = f"{name}_1.{ending}"
            # return
            self._print(f"File exist and will instead be written as {filename}")
            
        path = os.path.join(path, filename)


        if type(result) == np.ndarray or (type(result) == dict and type(result[list(result.keys())[0]]) == np.ndarray and (path.endswith('.npz') or path.endswith('.npy'))):
            if type(result) == np.ndarray and result.ndim < 3:
                np.savetxt(path, result)
            else:
                if path.endswith('.npz'):
                    np.savez_compressed(path, **result)
                else:
                    np.save(path, result, allow_pickle=False)
        
        elif type(result) == pd.DataFrame:
            result: pd.DataFrame
            pd.to_pickle(obj=result, filepath_or_buffer=path)
        
        elif filename.endswith('.yml') or filename.endswith('yaml'):
            with open(path, 'w') as stream:
                yaml.dump(data=result, stream=stream)
        
        elif filename.endswith('.csv'):
            with open(path, 'w+') as stream:
                writer = csv.writer(stream, delimiter='\t')
                writer.writerows(result)
        
        elif issubclass(type(result), torch.nn.Module):
            return self.save_model(model=result, filename=filename, path=path, overwrite=overwrite)
        
        else:
            with open(path, 'wb+') as f:
                dill.dump(result, f)

        self._print(f"Result successfully saved to {path}!")

    def load_result(self, filename:str, path:str=None, np_allow_pickle:bool=True):

        if path is None:
            path = self.root
        
        path = os.path.join(path, filename)

        if not os.path.exists(path):
            self._print(f"Result file {path} not found.")
            return None

        if path.endswith('.npy'):
            try:
                result = np.load(path, allow_pickle=np_allow_pickle)
            except pickle.UnpicklingError as e:
                result = np.loadtxt(path)


        elif filename.endswith('.yml') or filename.endswith('yaml'):
            with open(path, 'r') as stream:
                result = yaml.load(stream=stream, Loader=yaml.UnsafeLoader)
        else:
            with open(path, 'rb') as f:
               result = dill.load(f)

        return result
