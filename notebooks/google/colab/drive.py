import os

def mount(mountpoint, force_remount=False):
    print(f"Mocked Google Drive mount at: {mountpoint}")
    os.makedirs(mountpoint, exist_ok=True)
