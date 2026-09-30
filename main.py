import os
import sys
import subprocess


# ============================================
# Project Configuration
# ============================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_FOLDER = os.path.join(
    PROJECT_ROOT,
    "src"
)


# ============================================
# Run Python Script
# ============================================

def run_script(script_name):

    script_path = os.path.join(
        SRC_FOLDER,
        script_name
    )

    print("\n")
    print("=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            script_path
        ],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        print("\n")
        print("=" * 60)
        print(f"ERROR IN: {script_name}")
        print("=" * 60)

        print(
            "\nThe project stopped because "
            "this step failed."
        )

        sys.exit(
            result.returncode
        )

    print("\n")
    print("=" * 60)
    print(f"COMPLETED: {script_name}")
    print("=" * 60)


# ============================================
# Check Required Folders
# ============================================

def create_data_folder():

    data_folder = os.path.join(
        PROJECT_ROOT,
        "data"
    )

    os.makedirs(
        data_folder,
        exist_ok=True
    )


# ============================================
# Main Project Pipeline
# ============================================

def main():

    print("=" * 60)
    print("     SELF-SUPERVISED DOPPLER CLASSIFICATION")
    print("                  OTFS SYSTEM")
    print("=" * 60)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nPython executable:")
    print(sys.executable)

    create_data_folder()

    # ----------------------------------------
    # PART 1
    # Generate Dataset
    # ----------------------------------------

    print("\n")
    print("STEP 1: GENERATING OTFS DATASET")

    run_script(
        "dataset.py"
    )

    # ----------------------------------------
    # PART 2
    # Contrastive Learning
    # ----------------------------------------

    print("\n")
    print("STEP 2: SELF-SUPERVISED CONTRASTIVE LEARNING")

    run_script(
        "contrastive_train.py"
    )

    # ----------------------------------------
    # PART 3
    # Training + Validation
    # ----------------------------------------

    print("\n")
    print("STEP 3: TRAINING + VALIDATION")

    run_script(
        "training_curves.py"
    )

    # ----------------------------------------
    # PART 4
    # Train Doppler Classifier
    # ----------------------------------------

    print("\n")
    print("STEP 4: DOPPLER CLASSIFIER TRAINING")

    run_script(
        "classifier.py"
    )

    # ----------------------------------------
    # PART 5
    # Evaluate Model
    # ----------------------------------------

    print("\n")
    print("STEP 5: MODEL EVALUATION")

    run_script(
        "evaluation.py"
    )

    # ----------------------------------------
    # PART 6
    # Final Prediction
    # ----------------------------------------

    print("\n")
    print("STEP 6: FINAL DOPPLER PREDICTION")

    run_script(
        "predict.py"
    )

    # ----------------------------------------
    # Final Project Message
    # ----------------------------------------

    print("\n")
    print("=" * 60)
    print("          PROJECT COMPLETED")
    print("=" * 60)

    print("\nGenerated files:")

    print("\nDataset:")
    print("  data/signals.npy")
    print("  data/labels.npy")
    print("  data/velocities.npy")

    print("\nModels:")
    print("  data/contrastive_model.pth")
    print("  data/doppler_classifier.pth")

    print("\nEvaluation graphs:")
    print("  data/confusion_matrix.png")
    print("  data/prediction_distribution.png")
    print("  data/training_validation_loss.png")

    print("\nThe complete OTFS Doppler")
    print("classification pipeline has finished.")

    print("\n")
    print("=" * 60)


# ============================================
# Program Entry Point
# ============================================

if __name__ == "__main__":

    main()