"""This module provides a function to select a CUDA device and set environment variables.

Functions:
    cudaDeviceSelection(preselected_device: int = -1, device_order: str = "PCI_BUS_ID") -> None:
        Selects a CUDA device and sets environment variables. 
        Raises an exception if no CUDA device is detected.
    cudaInfo() -> str: Returns a string with information about the CUDA device(s).

"""

import torch, os

from deep_learning_lab import logging


_LOGGER = logging.getLogger(__name__)



def cudaDeviceSelection(preselected_device = None, device_order: str = "PCI_BUS_ID") -> int:
    """Select a CUDA device and set environment variables.

    Args:
        preselected_device: The index of the GPU device to use for computations. None promps the user and -1 is for CPU.
        device_order (str): The device order to use. Default is "PCI_BUS_ID".

    Raises:
        Exception: If no CUDA device is detected.

    Returns:
        int: The index of the selected device. -1 if it is the CPU.

    """
    # Do not select CUDA device if CUDA is not available
    if not torch.cuda.is_available():
        _LOGGER.warning("CUDA is not available. CPU selected")
        return -1
        
    # Get the number of CUDA devices available
    num_cuda_devices = torch.cuda.device_count()

    # Do not select CUDA device if no GPU is detected
    if num_cuda_devices == 0:
        _LOGGER.warning("Error: No CUDA device detected. Please ensure you have GPU devices on your infrastructure.")
        return -1
    
    # Select the preselected device if it is valid
    if preselected_device != None and -1 <= preselected_device < num_cuda_devices:
        selected_device = preselected_device
    else:
        # If the preselected device is not valid, print the list of available devices
        print(f"Device -1: CPU")
        for device in range(num_cuda_devices):
            print(f"Device {device}: {torch.cuda.get_device_properties(device)}")

        # Wait for the user to select a valid device
        valid = False
        while not valid:
            selected_device = int(input("Select a device:"))
            if -1 <= selected_device < num_cuda_devices:
                valid = True

    
    # if CPU has been selected, GPU setup is ignored
    if selected_device == -1:
        # Log CUDA information using the cudaInfo function
        _LOGGER.info(f"Infrastructure: %s", cudaInfo(selected_device).replace('\n', '; '))
        return selected_device
    
    # Set the CUDA environment variables
    os.environ["CUDA_DEVICE_ORDER"] = device_order
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_device)
    torch.cuda.set_device(selected_device)
    
    # Log CUDA information using the cudaInfo function
    _LOGGER.info("Infrastructure: %s", cudaInfo(selected_device).replace('\n', '; '))
    
    return selected_device
    

def cudaInfo(device) -> str:
    """Get information about the CUDA device(s).

    Returns:
        str: A message containing information about the CUDA device(s).

    """

    # type "nvidia-smi" on linux prompt for info about CUDA
    message = f"Torch ({torch.__version__})\n"
    message += f"CUDA ({torch.version.cuda})\n"
    if device == -1:
        message += "CPU"
    elif torch.cuda.is_available() and -1 < device < torch.cuda.device_count():
        message += f"GPU ({torch.cuda.get_device_properties(device).name})\n"
        message += f"Total CUDA memory ({torch.cuda.get_device_properties(device).total_memory//1024**2/1000:.2f} GB)\n"
        message += f"Allocated CUDA memory ({torch.cuda.memory_allocated()//1024**2/1000:.2f} GB)"
    return message
