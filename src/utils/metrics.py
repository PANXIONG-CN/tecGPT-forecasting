import torch
import numpy as np
from typing import Optional, Union

# TODO: Consider a fill_value or mask for all metrics to ignore certain values (e.g., padding or known missing data)
# Current implementation assumes y_true and y_pred are already filtered or that all values are valid.
# If a mask is needed, it should be a boolean tensor of the same shape as y_true/y_pred.

def calculate_mae(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算平均绝对误差 (Mean Absolute Error).
    Args:
        y_true: 真实值. Shape: [..., N_features_or_nodes_or_steps]
        y_pred: 预测值. Shape: [..., N_features_or_nodes_or_steps]
        mask: 可选的布尔掩码，为 True 的位置参与计算. Shape: same as y_true.
    Returns:
        torch.Tensor: MAE 值 (标量).
    """
    if mask is not None:
        if y_true.shape != mask.shape or y_pred.shape != mask.shape:
            raise ValueError("Mask shape must match y_true and y_pred shapes.")
        error = torch.abs(y_pred[mask] - y_true[mask])
        if error.numel() == 0: # All values masked out
            return torch.tensor(float('nan'), device=y_true.device, dtype=y_true.dtype)
        return torch.mean(error)
    else:
        return torch.mean(torch.abs(y_pred - y_true))

def calculate_rmse(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算均方根误差 (Root Mean Squared Error).
    Args:
        y_true: 真实值.
        y_pred: 预测值.
        mask: 可选掩码.
    Returns:
        torch.Tensor: RMSE 值 (标量).
    """
    if mask is not None:
        if y_true.shape != mask.shape or y_pred.shape != mask.shape:
            raise ValueError("Mask shape must match y_true and y_pred shapes.")
        squared_error = torch.square(y_pred[mask] - y_true[mask])
        if squared_error.numel() == 0:
            return torch.tensor(float('nan'), device=y_true.device, dtype=y_true.dtype)
        return torch.sqrt(torch.mean(squared_error))
    else:
        return torch.sqrt(torch.mean(torch.square(y_pred - y_true)))

def calculate_wmape(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-7 # To prevent division by zero if y_true is 0
) -> torch.Tensor:
    """
    计算加权平均绝对百分比误差 (Weighted Mean Absolute Percentage Error).
    WMAPE = sum(|y_true - y_pred|) / sum(|y_true|)
    Also known as MAD/Mean ratio or Coefficient of Variation of the RMSE if RMSE is used in numerator.
    Here, it's sum of absolute errors divided by sum of absolute true values.
    Args:
        y_true: 真实值.
        y_pred: 预测值.
        mask: 可选掩码.
        epsilon: 小常数避免除以零.
    Returns:
        torch.Tensor: WMAPE 值 (标量).
    """
    if mask is not None:
        if y_true.shape != mask.shape or y_pred.shape != mask.shape:
            raise ValueError("Mask shape must match y_true and y_pred shapes.")
        
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        
        if y_true_masked.numel() == 0: # All values masked out
            return torch.tensor(float('nan'), device=y_true.device, dtype=y_true.dtype)
            
        numerator = torch.sum(torch.abs(y_true_masked - y_pred_masked))
        denominator = torch.sum(torch.abs(y_true_masked))
    else:
        if y_true.numel() == 0:
            return torch.tensor(float('nan'), device=y_true.device, dtype=y_true.dtype)
        numerator = torch.sum(torch.abs(y_true - y_pred))
        denominator = torch.sum(torch.abs(y_true))

    if denominator < epsilon:
        # If sum of true values is close to zero, WMAPE can be very large or undefined.
        # Return NaN or a large number, or handle as per specific requirements.
        # For now, if denominator is effectively zero, but numerator is non-zero, result is inf.
        # If both are zero, result is 0/0 -> NaN.
        if numerator < epsilon: # Both are zero
            return torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
        else: # Non-zero error over zero sum of true values
            return torch.tensor(float('inf'), device=y_true.device, dtype=y_true.dtype)
            
    return numerator / denominator

def calculate_r_squared(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算 R 平方 (Coefficient of Determination).
    R² = 1 - (SS_res / SS_tot)
    SS_res = sum((y_true - y_pred)²)
    SS_tot = sum((y_true - mean(y_true))²)
    Args:
        y_true: 真实值.
        y_pred: 预测值.
        mask: 可选掩码.
    Returns:
        torch.Tensor: R² 值 (标量).
    """
    if mask is not None:
        if y_true.shape != mask.shape or y_pred.shape != mask.shape:
            raise ValueError("Mask shape must match y_true and y_pred shapes.")
            
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]

        if y_true_masked.numel() < 2: # Need at least 2 points for SS_tot to be meaningful
            return torch.tensor(float('nan'), device=y_true.device, dtype=y_true.dtype)
            
        ss_res = torch.sum(torch.square(y_true_masked - y_pred_masked))
        ss_tot = torch.sum(torch.square(y_true_masked - torch.mean(y_true_masked)))
    else:
        if y_true.numel() < 2:
            return torch.tensor(float('nan'), device=y_true.device, dtype=y_true.dtype)

        ss_res = torch.sum(torch.square(y_true - y_pred))
        ss_tot = torch.sum(torch.square(y_true - torch.mean(y_true)))

    if ss_tot < 1e-7: # Avoid division by zero if all y_true values are the same
        # If ss_tot is zero, it means all true values are identical.
        # If ss_res is also zero (perfect prediction), R² is undefined (conventionally 1 or NaN).
        # If ss_res is non-zero (imperfect prediction of a constant), R² is -infinity (or undefined).
        if ss_res < 1e-7: # Both zero
             return torch.tensor(1.0, device=y_true.device, dtype=y_true.dtype) # Or NaN by some conventions
        else:
             return torch.tensor(float('-inf'), device=y_true.device, dtype=y_true.dtype) # Indicates bad fit for constant data

    return 1 - (ss_res / ss_tot)


# Alias for R-squared if preferred (R is typically Pearson correlation for single dimension)
calculate_r2 = calculate_r_squared

# Example Usage:
if __name__ == '__main__':
    # Example Tensors
    y_true_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    y_pred_tensor = torch.tensor([1.1, 2.2, 2.8, 4.3, 4.8, 0.5])
    mask_tensor = torch.tensor([True, True, True, False, True, True]) # Mask out the 4th element
    fill_val = -1.0
    y_true_with_fill = torch.tensor([1.0, 2.0, fill_val, 4.0, 5.0])
    y_pred_with_fill = torch.tensor([1.1, 1.8, 0.0, 4.3, 4.8]) # prediction for fill_val is ignored
    mask_from_fill = (y_true_with_fill != fill_val)


    print("--- Without Mask ---")
    mae_val = calculate_mae(y_true_tensor, y_pred_tensor)
    rmse_val = calculate_rmse(y_true_tensor, y_pred_tensor)
    wmape_val = calculate_wmape(y_true_tensor, y_pred_tensor)
    r2_val = calculate_r_squared(y_true_tensor, y_pred_tensor)
    print(f"MAE: {mae_val.item()}")
    print(f"RMSE: {rmse_val.item()}")
    print(f"WMAPE: {wmape_val.item()}")
    print(f"R²: {r2_val.item()}")

    print("\n--- With Mask (masking out one element) ---")
    mae_masked = calculate_mae(y_true_tensor, y_pred_tensor, mask=mask_tensor)
    rmse_masked = calculate_rmse(y_true_tensor, y_pred_tensor, mask=mask_tensor)
    wmape_masked = calculate_wmape(y_true_tensor, y_pred_tensor, mask=mask_tensor)
    r2_masked = calculate_r_squared(y_true_tensor, y_pred_tensor, mask=mask_tensor)
    print(f"MAE (masked): {mae_masked.item()}")
    print(f"RMSE (masked): {rmse_masked.item()}")
    print(f"WMAPE (masked): {wmape_masked.item()}")
    print(f"R² (masked): {r2_masked.item()}")

    print("\n--- With Mask (from fill value) ---")
    mae_fill = calculate_mae(y_true_with_fill, y_pred_with_fill, mask=mask_from_fill)
    rmse_fill = calculate_rmse(y_true_with_fill, y_pred_with_fill, mask=mask_from_fill)
    wmape_fill = calculate_wmape(y_true_with_fill, y_pred_with_fill, mask=mask_from_fill)
    r2_fill = calculate_r_squared(y_true_with_fill, y_pred_with_fill, mask=mask_from_fill)
    print(f"MAE (fill_masked): {mae_fill.item()}")
    print(f"RMSE (fill_masked): {rmse_fill.item()}")
    print(f"WMAPE (fill_masked): {wmape_fill.item()}")
    print(f"R² (fill_masked): {r2_fill.item()}")
    
    # Edge cases for WMAPE and R2
    print("\n--- Edge Cases ---")
    # WMAPE with zero denominator
    y_true_zeros = torch.tensor([0.0, 0.0, 0.0])
    y_pred_non_zeros = torch.tensor([0.1, 0.2, 0.1])
    print(f"WMAPE (true_zeros, pred_non_zeros): {calculate_wmape(y_true_zeros, y_pred_non_zeros).item()}") # Should be inf
    print(f"WMAPE (true_zeros, pred_zeros): {calculate_wmape(y_true_zeros, y_true_zeros).item()}")       # Should be 0

    # R2 with constant true values
    y_true_const = torch.tensor([2.0, 2.0, 2.0])
    y_pred_perfect = torch.tensor([2.0, 2.0, 2.0])
    y_pred_imperfect = torch.tensor([2.1, 1.9, 2.0])
    print(f"R² (true_const, pred_perfect): {calculate_r_squared(y_true_const, y_pred_perfect).item()}") # Should be 1.0
    print(f"R² (true_const, pred_imperfect): {calculate_r_squared(y_true_const, y_pred_imperfect).item()}") # Should be -inf or large negative

    # R2 with only one data point (after mask)
    y_true_one_pt = torch.tensor([1.0, 2.0])
    y_pred_one_pt = torch.tensor([1.0, 2.0])
    mask_one_pt = torch.tensor([True, False])
    print(f"R² (one point after mask): {calculate_r_squared(y_true_one_pt, y_pred_one_pt, mask=mask_one_pt).item()}") # Should be NaN 