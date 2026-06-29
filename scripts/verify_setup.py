import os
import sys

def verify():
    print("=" * 60)
    print("PHAST PIPELINE SETUP VERIFICATION")
    print("=" * 60)
    
    # 1. Check directory junction
    colab_path = "/content/drive/MyDrive/exoplanet_pipeline"
    print(f"Checking Colab path junction ({colab_path})...")
    if os.path.exists(colab_path):
        print(f"[OK] Junction path is valid and resolves to: {os.path.abspath(colab_path)}")
        try:
            items = os.listdir(colab_path)
            print(f"[OK] Successfully listed directory contents (found {len(items)} items)")
        except Exception as e:
            print(f"[ERROR] Failed to list directory contents: {e}")
    else:
        print(f"[ERROR] Junction path does NOT exist. Expected at: {colab_path}")
        print("  Please check that the junction was created correctly.")
        
    # 2. Check mock google.colab package
    print("\nChecking mock google.colab package...")
    try:
        # Add workspace and notebooks to path just like the runner does
        workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        notebooks_dir = os.path.join(workspace_dir, "notebooks")
        if workspace_dir not in sys.path:
            sys.path.insert(0, workspace_dir)
        if notebooks_dir not in sys.path:
            sys.path.insert(0, notebooks_dir)
            
        from google.colab import drive
        drive.mount("/content/drive")
        print("[OK] Mock google.colab imported and executed successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to import/execute mock google.colab: {e}")
        
    # 3. Check requirements
    print("\nChecking exoplanet pipeline library dependencies...")
    libs = [
        ("lightkurve", "lightkurve"),
        ("celerite2", "celerite2"),
        ("transitleastsquares", "transitleastsquares"),
        ("xgboost", "xgboost"),
        ("batman", "batman-package"),
        ("torch", "torch"),
        ("emcee", "emcee"),
        ("corner", "corner"),
        ("astroquery", "astroquery"),
        ("tqdm", "tqdm")
    ]
    
    all_ok = True
    for import_name, package_name in libs:
        try:
            __import__(import_name)
            print(f"[OK] {package_name:20s}: installed")
        except ImportError:
            print(f"[MISSING] {package_name:20s}: NOT installed (import name: '{import_name}')")
            all_ok = False
            
    print("\n" + "=" * 60)
    if all_ok and os.path.exists(colab_path):
        print("VERIFICATION SUCCESSFUL: Everything is ready to run!")
    else:
        print("VERIFICATION FAILED: Please check setup.")
    print("=" * 60)

if __name__ == "__main__":
    verify()
