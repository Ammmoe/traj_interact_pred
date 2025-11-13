"""
Utility module for feature scaling in multi-agent trajectory data.

Provides helper functions to apply normalization or standardization
on a per-agent basis while preserving the original data structure.
"""


def scale_per_agent(data, scaler, features_per_agent, fit=False, inverse=False):
    """
    Scales or inverse-scales multi-agent feature data using a given scaler.

    The function reshapes the input so that scaling is applied consistently
    across all agents and samples, then restores the original shape.

    Args:
        data (np.ndarray): Input array of shape (num_samples, num_agents, features_per_agent).
        scaler: Scikit-learn scaler (e.g., MinMaxScaler, StandardScaler).
        features_per_agent (int): Number of features for each agent.
        fit (bool): If True, fits the scaler before transforming.
        inverse (bool): If True, applies inverse_transform instead of transform.

    Returns:
        np.ndarray: Scaled data with the same shape as input.
    """
    orig_shape = data.shape  # (num_samples, num_agents, features_per_agent)
    data_reshaped = data.reshape(
        -1, features_per_agent
    )  # (num_samples * num_agents, features_per_agent)

    if fit:
        scaler.fit(data_reshaped)

    if inverse:
        data_scaled = scaler.inverse_transform(data_reshaped)
    else:
        data_scaled = scaler.transform(data_reshaped)

    return data_scaled.reshape(orig_shape)
