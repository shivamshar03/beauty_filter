import sys
import traceback
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    import mediapipe as mp
    from math import atan2, degrees, sqrt
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"[ERROR] Missing required package: {e}")
    print("Please install required packages: pip install opencv-python mediapipe numpy scikit-learn")
    sys.exit(1)

def main():
    """Main entry point with comprehensive error handling"""
    try:
        app = SmartBeautyFilter()
        success = app.run()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n[INFO] Application interrupted by user")
        sys.exit(0)
    except CameraError as e:
        print(f"[ERROR] Camera error: {e}")
        print("[INFO] Please check camera connection and permissions")
        sys.exit(1)
    except ImportError as e:
        print(f"[ERROR] Missing required dependency: {e}")
        print("[INFO] Please install required packages: pip install opencv-python mediapipe numpy scikit-learn")
        sys.exit(1)
    except Exception as e:
        print(f"[CRITICAL] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()