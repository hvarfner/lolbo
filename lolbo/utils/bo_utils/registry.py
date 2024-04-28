from gpytorch.models import GP
from lolbo.utils.bo_utils.ppgpr import (
    GPModel, 
    GPModelDKL, 
    VanillaBOGPModel,
    VanillaBOZGPModel,
    ZGPModelDKL,
    ZGPModel,
    UnwhitenedVanillaBOZGPModel,
    UnwhitenedVanillaBOGPModel,
    ExactGPModel,
    ExactHenryModel,
    DeepGPModelDKL,
    ShallowGPModelDKL,
)


def get_model(gp_name: str) -> GP:
    if gp_name == "gp":
        return GPModel
    
    elif gp_name == "vanilla":
        return VanillaBOGPModel

    elif gp_name == "henry":
        return ExactHenryModel
        
    elif gp_name == "exact":
        return ExactGPModel
    
    elif gp_name == "dkl":
        return GPModelDKL
    
    elif gp_name == "deepdkl":
        return DeepGPModelDKL
    
    elif gp_name == "shallowdkl":
        return ShallowGPModelDKL
    
    elif gp_name == "zgp":
        return ZGPModel

    elif gp_name == "zvanilla":
        return VanillaBOZGPModel
    
    elif gp_name == "zdkl":
        return ZGPModelDKL
    
    elif gp_name == "unzvanilla":
        return UnwhitenedVanillaBOZGPModel 
       
    elif gp_name == "unvanilla":
        return UnwhitenedVanillaBOZGPModel
    
    else:
        raise ValueError(f"The GP model {gp_name} does not exist.")