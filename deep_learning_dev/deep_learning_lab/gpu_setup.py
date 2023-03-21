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



def cudaDeviceSelection(preselected_device: int = -1, device_order: str = "PCI_BUS_ID") -> None:
    """
    Select a CUDA device and set environment variables.

    Args:
        preselected_device (int): The index of the device to use. Default is -1.
        device_order (str): The device order to use. Default is "PCI_BUS_ID".

    Raises:
        Exception: If no CUDA device is detected.

    Returns:
        None
    """
    
    # Get the number of CUDA devices available
    num_cuda_devices = torch.cuda.device_count()

    # Raise an exception if no CUDA device is detected
    if num_cuda_devices == 0:
        msg = "Error: No CUDA device detected. Please ensure you have installed CUDA and have GPU devices on your infrastructure."
        _LOGGER.error(msg)
        raise Exception(msg)
    
    # Select the preselected device if it's valid
    if 0 <= preselected_device < num_cuda_devices:
        selected_device = preselected_device
    else:
        # If the preselected device is not valid, print the list of available devices
        for device in range(num_cuda_devices):
            print(f"Device {device}: {torch.cuda.get_device_properties(device)}")

        # Wait for the user to select a valid device
        valid = False
        while not valid:
            selected_device = input("Select a device:")
            if 0 <= selected_device < num_cuda_devices:
                valid = True
        _LOGGER.info(f"User selected {selected_device} device")


    # Set the CUDA environment variables
    os.environ["CUDA_DEVICE_ORDER"] = device_order
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_device)
    torch.cuda.set_device(selected_device)
    
    # Log CUDA information using the cudaInfo function
    _LOGGER.info("Infrastructure: %s", cudaInfo().replace('\n', '; '))
    

def cudaInfo() -> str:
    """
    Get information about the CUDA device(s).

    Returns:
        str: A message containing information about the CUDA device(s).
    """
    # type "nvidia-smi" on linux prompt for info about CUDA
    message = f"Torch ({torch.__version__})\n"
    message += f"CUDA ({torch.version.cuda})\n"
    message += f"GPU ({torch.cuda.get_device_properties(0).name})\n"
    message += f"CUDA memory ({torch.cuda.get_device_properties(0).total_memory//1024**2/1000:.2f} GB)"
    return message
