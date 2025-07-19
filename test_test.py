import subprocess
import torch
import pynvml
#It can be used to obtain GPU runtime information, such as power consumption, power limitations, memory usage, etc

pynvml.nvmlInit()
print("Torch:", torch.__version__)
#Output the PyTorch version

print("CUDA runtime version:", torch.version.cuda)
#Output the CUDA runtime version supported by PyTorch

print("Is CUDA available?", torch.cuda.is_available())
#Determine if there is a usable NVIDIA graphics card

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))    #Output the name of the GPU
driver = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    text=True
).strip()   #Query the NVIDIA driver version

handle = pynvml.nvmlDeviceGetHandleByIndex(0)

try:
    power_cap = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
    #Try to read the upper limit of the graphics card's power consumption (in watts)
except pynvml.NVMLError:
    power_cap = None

cap_txt = f"{power_cap:.0f} W" if power_cap else "N/A"

print(f"Driver: {driver}  |  Power-limit: {cap_txt}")
#Print the NVIDIA driver version and power consumption limit

mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Used memory: {mem_info.used / 1024**2:.0f} MB")
print(f"Free memory: {mem_info.free / 1024**2:.0f} MB")
print(f"Total GPU Memory: {mem_info.total / 1024**2:.0f} MB")
#Print Used memory,Free memory, maximum memory

