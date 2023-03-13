import torch
import os


def cudaDeviceSelection(preselected_device= -1, device_order= "PCI_BUS_ID"):
    num_cuda_devices = torch.cuda.device_count()
    if num_cuda_devices == 0:
        raise Exception("Error: No CUDA device detected. Please ensure you have installed CUDA and have GPU devices on your infrastructure.")
    
    if 0 <= preselected_device < num_cuda_devices:
        selected_device = preselected_device
    else:
        for device in range(num_cuda_devices):
            print(f"Device {device}: {torch.cuda.get_device_properties(device)}")
        valid = False
        while not valid:
            selected_device = input("Select a device:")
            if 0 <= selected_device < num_cuda_devices:
                valid = True

    os.environ["CUDA_DEVICE_ORDER"] = device_order
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_device)
    torch.cuda.set_device(selected_device)


def cudaInfo():
    # type "nvidia-smi" on linux prompt for info about CUDA
    message = "Torch: " + str(torch.__version__) + '\n'
    message += "CUDA: " + str(torch.version.cuda) + '\n'
    message += "Device: " + str(torch.cuda.get_device_properties(0).name) + '\n'
    message += "Memory: %.2f GB" % (torch.cuda.get_device_properties(0).total_memory//1024**2/1000)
    print(message)
