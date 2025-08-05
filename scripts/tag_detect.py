import subprocess

try:
    import pynvml
except Exception:
    pynvml = None

def _by_nvml():
    if pynvml is None:
        return None, "pynvml_not_installed"
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(h)
        limit_w = limit_mw / 1000.0
        pynvml.nvmlShutdown()
        return limit_w, "ok"
    except Exception as e:
        return None, f"nvml_error:{type(e).__name__}:{e}"

def _by_nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q", "-d", "POWER"],
            stderr=subprocess.STDOUT, text=True, timeout=5
        )
        for line in out.splitlines():
            if "Power Limit" in line and "W" in line:
                val = line.split(":")[1].strip().split()[0]
                return float(val), "smi_ok"
        return None, "smi_parse_fail"
    except Exception as e:
        return None, f"smi_error:{type(e).__name__}:{e}"

def detect_tag_by_nvml(threshold=90):
    limit_w, why = _by_nvml()
    if limit_w is None:
        limit_w, why2 = _by_nvidia_smi()
        if limit_w is None:
            return None, f"{why}|{why2}"
    tag = "65W" if limit_w <= threshold else "115W"
    return tag, limit_w
