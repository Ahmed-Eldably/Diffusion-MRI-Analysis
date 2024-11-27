import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import pandas as pd
from scipy.optimize import curve_fit



def read_dicom_data_with_directions(base_dir, b_values, dataset_folders):
    """
    Reads DICOM data from specified dataset folders, associates files with b-values and directions.
    """
    datasets = {}

    for dataset_name, folder_map in dataset_folders.items():
        dataset_data = []
        for direction, subdirs in folder_map.items():
            for subdir in subdirs:
                folder_path = os.path.join(base_dir, subdir)
                dicom_files = sorted(
                    [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".dcm")]
                )

                if len(dicom_files) != len(b_values):
                    print(f"Warning: Number of DICOM files ({len(dicom_files)}) does not match b-values ({len(b_values)}) for {dataset_name}/{direction}/{subdir}")
                    continue

                for file, b_value in zip(dicom_files, b_values):
                    ds = pydicom.dcmread(file)
                    dataset_data.append({
                        "file": file,
                        "b_value": b_value,
                        "direction": direction,
                        "data": ds.pixel_array,
                        "md": ds
                    })

        datasets[dataset_name] = dataset_data

    return datasets

def get_circular_roi(image, center, radius):
    """
    Extract a circular ROI from an image.
    """
    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    roi = image[mask]
    return roi, mask

def process_single_image(image, centers, radii):
    """
    Process a single DICOM image to extract ROI metrics.
    """
    results = []
    for idx, (center, radius) in enumerate(zip(centers, radii), start=1):
        roi, mask = get_circular_roi(image, center, radius)
        mean_intensity = np.mean(roi)
        std_intensity = np.std(roi)

        results.append({
            "ROI": idx,
            "Mean Intensity": mean_intensity,
            "Std Dev": std_intensity
        })
    return results

def process_image_stack_with_bvalues(dataset_data, centers, radii):
    """
    Process a stack of images for a dataset to extract ROI metrics.
    """
    results = []
    for entry in dataset_data:
        b_value = entry["b_value"]
        direction = entry["direction"]
        image = entry["data"]

        roi_metrics = process_single_image(image, centers, radii)
        for metric in roi_metrics:
            results.append({
                "b_value": b_value,
                "direction": direction,
                "ROI": metric["ROI"],
                "Mean Intensity": metric["Mean Intensity"],
                "Std Dev": metric["Std Dev"]
            })

    return pd.DataFrame(results)

def visualize_image_with_rois(image, centers, radii, title="Image with ROI(s)", save_path=None, show=True):
    """
    Visualize a single DICOM image with one or more ROIs overlaid.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.title(title, fontsize=14)

    # Define colors for ROIs
    colors = ["red", "yellow", "green", "blue", "cyan"]

    # Overlay ROIs
    for idx, (center, radius) in enumerate(zip(centers, radii)):
        _, mask = get_circular_roi(image, center, radius)
        plt.contour(mask, colors=[colors[idx % len(colors)]], linewidths=1.5, label=f"ROI {idx + 1}")

    plt.legend()
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory if not showing


def visualize_dataset_with_rois(dataset_data, centers, radii, dataset_name, save_path=None, show=False):
    """
    Visualize multiple images from a dataset with ROIs overlaid.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    for entry in dataset_data:
        b_value = entry["b_value"]
        direction = entry["direction"]
        image = entry["data"]

        # Generate title for the visualization
        title = f"{dataset_name.capitalize()} - {direction} (b={b_value})"

        # Generate save path if needed
        file_save_path = None
        if save_path:
            file_save_path = os.path.join(save_path, f"{dataset_name}_{direction}_b{b_value}.png")

        # Visualize the image with ROIs
        visualize_image_with_rois(image, centers, radii, title=title, save_path=file_save_path, show=show)


def plot_signal_decay_per_direction(results_df, dataset_name, direction, log_scale=True, save_path=None, show=True):
    """
    Plot signal intensity decay for all ROIs in a single dataset and direction.

    Parameters:
        results_df (DataFrame): DataFrame containing signal decay metrics.
        dataset_name (str): Name of the dataset (e.g., "healthy").
        direction (str): Diffusion encoding direction (e.g., "read", "phase", "slice").
        log_scale (bool): If True, use log scale for signal intensity (Y-axis).
        save_path (str): File path to save the plot (optional).
        show (bool): If True, display the plot interactively.
    """
    # Filter the DataFrame for the specified dataset and direction
    filtered_df = results_df[
        (results_df["Dataset"] == dataset_name) & (results_df["Direction"] == direction)
    ]

    if filtered_df.empty:
        print(f"No data available for dataset '{dataset_name}' and direction '{direction}'.")
        return

    plt.figure(figsize=(8, 6))
    plt.title(f"Signal Decay: {dataset_name.capitalize()} - {direction}", fontsize=14)

    # Plot signal decay for each ROI
    for roi_id in filtered_df["ROI"].unique():
        roi_data = filtered_df[filtered_df["ROI"] == roi_id]
        plt.plot(
            roi_data["B-Value"],  # X-axis
            roi_data["Mean Intensity"],  # Y-axis
            'o-', label=f"ROI {roi_id}"  # Legend label
        )

    # Configure axes
    plt.xlabel("b-value (s/mm²)", fontsize=12)
    if log_scale:
        plt.yscale("log")
        plt.ylabel("Log Mean Intensity", fontsize=12)
    else:
        plt.ylabel("Mean Intensity", fontsize=12)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_signal_decay_comparison(results_df, direction, log_scale=True, save_path=None, show=True):
    """
    Plot signal decay for all datasets and their ROIs in a single figure with subplots.

    Parameters:
        results_df (DataFrame): DataFrame containing signal decay metrics.
        direction (str): Diffusion encoding direction (e.g., "read", "phase", "slice").
        log_scale (bool): If True, use log scale for signal intensity (Y-axis).
        save_path (str): File path to save the plot (optional).
        show (bool): If True, display the plot interactively.
    """
    # Get unique datasets
    datasets = results_df["Dataset"].unique()
    num_datasets = len(datasets)

    # Create subplots
    fig, axes = plt.subplots(1, num_datasets, figsize=(5 * num_datasets, 6), sharey=True)
    if num_datasets == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    for ax, dataset in zip(axes, datasets):
        # Filter for dataset and direction
        filtered_df = results_df[
            (results_df["Dataset"] == dataset) & (results_df["Direction"] == direction)
        ]

        if filtered_df.empty:
            ax.set_title(f"{dataset.capitalize()} - No Data")
            continue

        # Plot signal decay for each ROI
        for roi_id in filtered_df["ROI"].unique():
            roi_data = filtered_df[filtered_df["ROI"] == roi_id]
            ax.plot(
                roi_data["B-Value"],
                roi_data["Mean Intensity"],
                'o-', label=f"ROI {roi_id}"
            )

        # Customize plot
        ax.set_title(f"{dataset.capitalize()} - {direction}", fontsize=14)
        ax.set_xlabel("b-value (s/mm²)", fontsize=12)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()

    # Set shared Y-axis label
    fig.text(0.04, 0.5, "Log Mean Intensity" if log_scale else "Mean Intensity",
             va="center", rotation="vertical", fontsize=12)

    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def save_signal_decay_plots(results_df, log_scale=True, save_dir="signal_decay_plots"):
    """
    Save signal decay plots for all datasets and directions into individual files.

    Parameters:
        results_df (DataFrame): DataFrame containing signal decay metrics.
        log_scale (bool): If True, use log scale for signal intensity (Y-axis).
        save_dir (str): Directory to save the plots.
    """
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get unique datasets and directions
    datasets = results_df["Dataset"].unique()
    directions = results_df["Direction"].unique()

    for dataset in datasets:
        for direction in directions:
            # Filter for dataset and direction
            filtered_df = results_df[
                (results_df["Dataset"] == dataset) & (results_df["Direction"] == direction)
            ]

            # Skip if no data is available for this combination
            if filtered_df.empty:
                print(f"Skipping: No data for {dataset} - {direction}")
                continue

            # Create a new figure
            plt.figure(figsize=(6, 6))

            # Plot signal decay for each ROI
            for roi_id in filtered_df["ROI"].unique():
                roi_data = filtered_df[filtered_df["ROI"] == roi_id]
                plt.plot(
                    roi_data["B-Value"],
                    roi_data["Mean Intensity"],
                    'o-', label=f"ROI {roi_id}"
                )

            # Customize plot
            plt.title(f"{dataset.capitalize()} - {direction}", fontsize=14)
            plt.xlabel("b-value (s/mm²)", fontsize=12)
            plt.ylabel("Log Mean Intensity" if log_scale else "Mean Intensity", fontsize=12)
            if log_scale:
                plt.yscale("log")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend(fontsize=10)

            # Save the plot
            save_path = os.path.join(save_dir, f"{dataset}_{direction}_signal_decay.png")
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved: {save_path}")

            # Close the figure to free memory
            plt.close()

def save_side_by_side_signal_decay_plots(results_df, log_scale=True, save_dir="side_by_side_plots"):
    """
    Save side-by-side signal decay plots for all datasets in a single figure for each direction.

    Parameters:
        results_df (DataFrame): DataFrame containing signal decay metrics.
        log_scale (bool): If True, use log scale for signal intensity (Y-axis).
        save_dir (str): Directory to save the plots.
    """
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get unique directions
    directions = results_df["Direction"].unique()

    for direction in directions:
        # Create a figure with subplots for each dataset
        datasets = results_df["Dataset"].unique()
        num_datasets = len(datasets)
        fig, axes = plt.subplots(1, num_datasets, figsize=(5 * num_datasets, 6), sharey=True)

        if num_datasets == 1:
            axes = [axes]  # Ensure axes is iterable for a single subplot

        for ax, dataset in zip(axes, datasets):
            # Filter for dataset and direction
            filtered_df = results_df[
                (results_df["Dataset"] == dataset) & (results_df["Direction"] == direction)
            ]

            if filtered_df.empty:
                ax.set_title(f"{dataset.capitalize()} - No Data")
                ax.axis("off")
                continue

            # Plot signal decay for each ROI
            for roi_id in filtered_df["ROI"].unique():
                roi_data = filtered_df[filtered_df["ROI"] == roi_id]
                ax.plot(
                    roi_data["B-Value"],
                    roi_data["Mean Intensity"],
                    'o-', label=f"ROI {roi_id}"
                )

            # Customize subplot
            ax.set_title(f"{dataset.capitalize()}", fontsize=14)
            ax.set_xlabel("b-value (s/mm²)", fontsize=12)
            if log_scale:
                ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.legend(fontsize=10)

        # Set shared Y-axis label
        fig.text(0.04, 0.5, "Log Mean Intensity" if log_scale else "Mean Intensity",
                 va="center", rotation="vertical", fontsize=12)

        # Save the figure
        save_path = os.path.join(save_dir, f"signal_decay_{direction}_side_by_side.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Side-by-side plot saved: {save_path}")

        # Close the figure to free memory
        plt.close()





def visualize_datasets_side_by_side(datasets, roi_params, titles, save_path=None, show=True):
    """
    Visualize the first image from multiple datasets side by side with ROIs.
    """
    # Create subplots with fixed size
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6))  # Fixed figure size for consistency

    # Iterate through datasets and plot each
    for idx, (dataset_name, dataset_data) in enumerate(datasets.items()):
        ax = axes[idx]
        entry = dataset_data[0]  # First image in the dataset
        image = entry["data"]
        centers = roi_params[dataset_name]["centers"]
        radii = roi_params[dataset_name]["radii"]

        # Plot the image
        ax.imshow(image, cmap="gray")
        ax.set_title(titles[idx], fontsize=14)
        ax.axis("off")

        # Overlay ROIs
        colors = ["red", "yellow", "green", "blue", "cyan"]
        for i, (center, radius) in enumerate(zip(centers, radii)):
            _, mask = get_circular_roi(image, center, radius)
            ax.contour(mask, colors=[colors[i % len(colors)]], linewidths=1.5)

    # Tight layout for spacing
    plt.tight_layout()

    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

def prepare_signal_decay_dataframe(dataset_data, centers, radii, dataset_name):
    """
    Prepare a DataFrame for signal decay plotting with specified columns.

    Parameters:
        dataset_data (list of dict): Dataset data loaded by `read_dicom_data_with_directions`.
        centers (list of tuples): List of ROI center coordinates.
        radii (list of int): List of ROI radii.
        dataset_name (str): Name of the dataset (e.g., "healthy").

    Returns:
        DataFrame: DataFrame with columns ['File', 'ROI', 'Image Number',
                                          'Mean Intensity', 'Std Dev',
                                          'B-Value', 'Dataset', 'Direction'].
    """
    results = []

    for image_idx, entry in enumerate(dataset_data, start=1):
        file = entry["file"]  # File path or name
        b_value = entry["b_value"]
        direction = entry.get("direction", None)  # Direction (if available)
        image = entry["data"]

        # Extract ROI metrics
        roi_metrics = process_single_image(image, centers, radii)

        for metric in roi_metrics:
            results.append({
                "File": file,
                "ROI": metric["ROI"],
                "Image Number": image_idx,  # Sequential number in the dataset
                "Mean Intensity": metric["Mean Intensity"],
                "Std Dev": metric["Std Dev"],
                "B-Value": b_value,
                "Dataset": dataset_name,
                "Direction": direction
            })

    # Convert results to a DataFrame with specified column order
    df = pd.DataFrame(results)
    return df


def compute_adc_values_ranges(metrics_df):
    """
    Compute ADC values for specified b-value ranges: all, low, and high.

    Parameters:
        metrics_df (DataFrame): DataFrame with signal intensities.

    Returns:
        DataFrame: ADC values for all, low, and high b-value ranges.
    """

    def signal_decay(b, S0, ADC):
        """Exponential decay function."""
        return S0 * np.exp(-b * ADC)

    # Define b-value ranges
    b_value_ranges = {
        "all": (0, 1000),  # ADC_all: All b-values
        "low": (100, 500),  # ADC_low: Low b-values
        "high": (600, 1000)  # ADC_high: High b-values
    }

    results = []
    grouped = metrics_df.groupby(["Dataset", "ROI", "Direction"])

    for (dataset, roi, direction), group in grouped:
        adc_result = {"Dataset": dataset, "ROI": roi, "Direction": direction}

        for key, (b_min, b_max) in b_value_ranges.items():
            # Filter data for the current b-value range
            b_group = group[(group["B-Value"] >= b_min) & (group["B-Value"] <= b_max)]

            if b_group.empty:
                adc_result[f"S0_{key}"] = np.nan
                adc_result[f"ADC_{key}"] = np.nan
                continue

            b_values = b_group["B-Value"].values
            intensities = b_group["Mean Intensity"].values

            try:
                # Fit the signal decay curve
                params, _ = curve_fit(signal_decay, b_values, intensities, p0=(intensities.max(), 0.001))
                adc_result[f"S0_{key}"] = params[0]
                adc_result[f"ADC_{key}"] = params[1]
            except RuntimeError:
                # Handle curve fitting failures
                adc_result[f"S0_{key}"] = np.nan
                adc_result[f"ADC_{key}"] = np.nan

        # Append results for this ROI and direction
        results.append(adc_result)

    return pd.DataFrame(results)


def plot_adc_results_rowwise(adc_df, save_path=None, show=True):
    """
    Improved visualization of ADC results row-wise for all datasets, with enhancements.

    Parameters:
        adc_df (DataFrame): DataFrame containing ADC results.
        save_path (str): File path to save the figure (optional).
        show (bool): If True, display the figure interactively.
    """
    # Sort datasets so isotropic appears first
    datasets = ["isotropic", "healthy", "leek"]
    directions = adc_df["Direction"].unique()
    adc_types = ["ADC_all", "ADC_low", "ADC_high"]

    # Create a figure with a row for each dataset and columns for directions
    fig, axes = plt.subplots(len(datasets), len(directions), figsize=(15, 5 * len(datasets)), sharey=False)
    axes = np.atleast_2d(axes)  # Ensure axes are 2D for single dataset cases

    for row_idx, dataset in enumerate(datasets):
        for col_idx, direction in enumerate(directions):
            ax = axes[row_idx, col_idx]
            # Filter data for the current dataset and direction
            filtered_df = adc_df[(adc_df["Dataset"] == dataset) & (adc_df["Direction"] == direction)]

            if filtered_df.empty:
                # Handle missing data
                ax.set_title(f"No data for {dataset.capitalize()} - {direction}", fontsize=10)
                ax.axis("off")
                continue

            # Bar plot for ADC values
            x = filtered_df["ROI"]
            width = 0.2
            x_positions = np.arange(len(x))
            bars_all = ax.bar(x_positions - width, filtered_df["ADC_all"], width, label="ADC_all", color="blue")
            bars_low = ax.bar(x_positions, filtered_df["ADC_low"], width, label="ADC_low", color="orange")
            bars_high = ax.bar(x_positions + width, filtered_df["ADC_high"], width, label="ADC_high", color="green")

            # Customize subplot
            ax.set_title(f"{dataset.capitalize()} - {direction}", fontsize=12)
            ax.set_xlabel("ROI", fontsize=10)
            if col_idx == 0:  # Add Y-axis label only for the first column
                ax.set_ylabel("ADC Values (mm²/s)", fontsize=10)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f"ROI {int(roi)}" for roi in x], fontsize=9)
            ax.grid(axis="y", linestyle="--", linewidth=0.5)

            # Highlight significant differences (optional): e.g., ADC_low > ADC_high
            for bar_low, bar_high in zip(bars_low, bars_high):
                if bar_low.get_height() > bar_high.get_height():  # Add marker for significant difference
                    ax.text(
                        bar_low.get_x() + bar_low.get_width() / 2,
                        bar_low.get_height() + 0.0001,
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="red"
                    )

            if row_idx == 0 and col_idx == len(directions) - 1:
                ax.legend(loc="upper right", fontsize=9)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def read_adc_maps(base_dir, adc_folders):
    """
    Reads ADC maps from specified ADC directories.

    Parameters:
        base_dir (str): Base directory containing ADC map datasets.
        adc_folders (dict): Mapping of dataset names to ADC subdirectories.

    Returns:
        dict: Organized ADC map data for each dataset.
    """
    adc_data = {}

    for dataset_name, folder_map in adc_folders.items():
        dataset_adc = []
        for direction, subdirs in folder_map.items():
            for subdir in subdirs:
                folder_path = os.path.join(base_dir, subdir)
                adc_files = sorted(
                    [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".dcm")]
                )
                if not adc_files:
                    print(f"Warning: No ADC map found for {dataset_name}/{direction}/{subdir}")
                    continue

                # Assuming one ADC map file per direction
                adc_file = adc_files[0]
                ds = pydicom.dcmread(adc_file)
                dataset_adc.append({
                    "file": adc_file,
                    "direction": direction,
                    "data": ds.pixel_array,
                })

        adc_data[dataset_name] = dataset_adc

    return adc_data


def extract_adc_map_values(adc_data, centers, radii):
    """
    Extract and scale ADC values from ADC maps in the dataset.

    Parameters:
        adc_data (list of dict): Loaded ADC map dataset.
        centers (list of tuples): List of ROI centers.
        radii (list of int): List of ROI radii.

    Returns:
        DataFrame: Extracted ADC values for each ROI, scaled to mm²/s.
    """
    results = []

    for entry in adc_data:
        # Process the ADC map image
        image = entry["data"]
        direction = entry["direction"]

        for idx, (center, radius) in enumerate(zip(centers, radii), start=1):
            roi, _ = get_circular_roi(image, center, radius)

            # Scale ADC values from micrometers²/s to millimeters²/s
            mean_adc = roi.mean() / 1e6
            std_adc = roi.std() / 1e6

            results.append({
                "ROI": idx,
                "Direction": direction,
                "Mean ADC": mean_adc,
                "Std Dev ADC": std_adc
            })

    return pd.DataFrame(results)


