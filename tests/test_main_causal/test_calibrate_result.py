import os
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

def test_calibration_results(
    finetune_dir: str,
    calibrated_dir: str,
    n_bins: int = 10,
    plot: bool = False
):
    """
    Test the calibration results by comparing original and calibrated predictions.
    
    Args:
        finetune_dir: Directory containing original finetune outputs
        calibrated_dir: Directory containing calibrated predictions
        n_bins: Number of bins for calibration curve
        plot: Whether to generate calibration plots
    """
    # Load original predictions
    original_preds = pd.read_csv(join(finetune_dir, "mock_predictions_and_targets.csv"))
    
    # Load calibrated predictions
    calibrated_preds = pd.read_csv(join(calibrated_dir, "predictions_and_targets_calibrated.csv"))
    
    # 1. Basic Validation Tests
    print("\n=== Basic Validation Tests ===")
    
    # Check that we have the same subjects
    assert set(original_preds['subject_id']) == set(calibrated_preds['subject_id']), \
        "Subject IDs don't match between original and calibrated predictions"
    print("✓ Subject IDs match between original and calibrated predictions")
    
    # Check probability ranges
    assert (calibrated_preds['probas'] >= 0).all() and (calibrated_preds['probas'] <= 1).all(), \
        "Calibrated probabilities outside [0,1] range"
    print("✓ Calibrated probabilities are within [0,1] range")
    
    # Check targets remain unchanged
    merged_df = pd.merge(
        original_preds, 
        calibrated_preds, 
        on='subject_id', 
        suffixes=('_orig', '_cal')
    )
    assert (merged_df['targets_orig'] == merged_df['targets_cal']).all(), \
        "Targets changed during calibration"
    print("✓ Targets unchanged by calibration")
    
    # 2. Calibration Quality Tests
    print("\n=== Calibration Quality Tests ===")
    
    # Compute calibration curves
    prob_true_orig, prob_pred_orig = calibration_curve(
        merged_df['targets_orig'], 
        merged_df['probas_orig'], 
        n_bins=n_bins
    )
    
    prob_true_cal, prob_pred_cal = calibration_curve(
        merged_df['targets_cal'], 
        merged_df['probas_cal'], 
        n_bins=n_bins
    )
    
    # Compute calibration error (mean absolute difference between predicted and true probabilities)
    cal_error_orig = np.mean(np.abs(prob_pred_orig - prob_true_orig))
    cal_error_cal = np.mean(np.abs(prob_pred_cal - prob_true_cal))
    
    print(f"Original calibration error: {cal_error_orig:.4f}")
    print(f"Calibrated calibration error: {cal_error_cal:.4f}")
    
    # Compute Brier scores
    brier_orig = brier_score_loss(merged_df['targets_orig'], merged_df['probas_orig'])
    brier_cal = brier_score_loss(merged_df['targets_cal'], merged_df['probas_cal'])
    
    print(f"Original Brier score: {brier_orig:.4f}")
    print(f"Calibrated Brier score: {brier_cal:.4f}")
    
    # Compute AUC scores (should remain similar)
    auc_orig = roc_auc_score(merged_df['targets_orig'], merged_df['probas_orig'])
    auc_cal = roc_auc_score(merged_df['targets_cal'], merged_df['probas_cal'])
    
    print(f"Original AUC: {auc_orig:.4f}")
    print(f"Calibrated AUC: {auc_cal:.4f}")
    
    # Optional: Generate calibration plots
    if plot:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 10))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(prob_pred_orig, prob_true_orig, 'ro-', label='Original')
        plt.plot(prob_pred_cal, prob_true_cal, 'bo-', label='Calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(join(calibrated_dir, 'calibration_curves.png'))
        plt.close()
        
        print("\n✓ Calibration plot saved")
    
    # Return summary statistics
    return {
        'calibration_error': {'original': cal_error_orig, 'calibrated': cal_error_cal},
        'brier_score': {'original': brier_orig, 'calibrated': brier_cal},
        'auc': {'original': auc_orig, 'calibrated': auc_cal}
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test calibration results")
    parser.add_argument(
        "--finetune_dir",
        type=str,
        default="./outputs/generated/finetune",
        help="Directory containing original finetune outputs"
    )
    parser.add_argument(
        "--calibrated_dir",
        type=str,
        default="./outputs/generated/calibrated_predictions",
        help="Directory containing calibrated predictions"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate calibration plots"
    )
    
    args = parser.parse_args()
    
    results = test_calibration_results(
        args.finetune_dir,
        args.calibrated_dir,
        plot=args.plot
    )
    
    # Print summary
    print("\n=== Summary ===")
    print("Calibration improved:" if results['calibration_error']['calibrated'] < results['calibration_error']['original'] 
          else "Calibration did not improve")
    print("Brier score improved:" if results['brier_score']['calibrated'] < results['brier_score']['original']
          else "Brier score did not improve")
    print(f"AUC change: {results['auc']['calibrated'] - results['auc']['original']:.4f}")
 