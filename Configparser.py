import yaml
import csv
import sys


def yamlConfigParser(filepath:str='./ToolConfig.yaml'):
    if filepath.split('.')[-1] !='yaml':
        sys.exit("YAML file expected.")
    else:
        with open(filepath, 'r') as file:
            configDict = yaml.safe_load(file)
            argList =  [configDict['MR-array-size'], configDict['MR-radius'], configDict['MR-gap-from-waveguide'], configDict[ 'inter-MR-distance'],
                        configDict['Waveguide-width'], configDict['Distance-between-waveguides'], configDict['waveguide-Pattern'], configDict['polynomial-degree']]
            Φ_input = csvReader(configDict['Φ_input'])
            distance = csvReader(configDict['distance'])
            δΦ = csvReader(configDict['δΦ'])

            return argList, Φ_input, distance, δΦ
        

def csvReader(filePath:str):
    if filePath!=None and filePath.split('.')[-1]=='csv':
        data =[]
        # TODO: CSV read implementation
        return data
    else:
        return None

