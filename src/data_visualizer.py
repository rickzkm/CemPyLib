"""
Visualization functions for meteor data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import logging
from datetime import datetime
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from src.config import PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set default style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")


def create_output_dir(subdir: str = 'visualizations') -> str:
    """
    Create an output directory for visualizations.
    
    Args:
        subdir: Subdirectory name
        
    Returns:
        Path to the output directory
    """
    output_dir = os.path.join(PROCESSED_DATA_DIR, subdir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_meteor_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> str:
    """
    Plot the distribution of meteors across the sky (RA/Dec).
    
    Args:
        df: Input DataFrame
        save_path: Path to save the plot (if None, use default location)
        
    Returns:
        Path to the saved plot
    """
    if '_ra_o' not in df.columns or '_dc_o' not in df.columns:
        logger.warning("Required columns '_ra_o' and/or '_dc_o' not found")
        return None
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot of RA/Dec
    plt.scatter(df['_ra_o'], df['_dc_o'], alpha=0.3, s=5)
    
    # Add colorbar by magnitude if available
    if '_amag' in df.columns:
        scatter = plt.scatter(
            df['_ra_o'], df['_dc_o'],
            c=df['_amag'],
            cmap='viridis',
            alpha=0.5,
            s=10
        )
        plt.colorbar(scatter, label='Apparent Magnitude')
    
    # Set labels and title
    plt.xlabel('Right Ascension (degrees)')
    plt.ylabel('Declination (degrees)')
    plt.title('Distribution of Meteors in the Sky')
    
    # Invert x-axis for RA (astronomical convention)
    plt.gca().invert_xaxis()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    if save_path is None:
        output_dir = create_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f'meteor_distribution_{timestamp}.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved meteor distribution plot to {save_path}")
    return save_path


def plot_orbital_elements(df: pd.DataFrame, save_path: Optional[str] = None) -> str:
    """
    Plot orbital elements of meteors.
    
    Args:
        df: Input DataFrame
        save_path: Path to save the plot (if None, use default location)
        
    Returns:
        Path to the saved plot
    """
    # Check required columns
    orbital_elements = ['_a', '_e', '_q', '_incl', '_peri']
    available_elements = [col for col in orbital_elements if col in df.columns]
    
    if len(available_elements) < 2:
        logger.warning("Not enough orbital elements available for plotting")
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2)
    
    # Plot 1: Semi-major axis (a) vs. Eccentricity (e)
    if '_a' in df.columns and '_e' in df.columns:
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Filter out extreme values
        plot_data = df[(df['_a'] > 0) & (df['_a'] < 100) & (df['_e'] >= 0) & (df['_e'] <= 1.5)]
        
        scatter = ax1.scatter(
            plot_data['_a'], 
            plot_data['_e'], 
            c=plot_data.get('_incl', 'blue'),
            cmap='viridis',
            alpha=0.5,
            s=5
        )
        
        if '_incl' in df.columns:
            plt.colorbar(scatter, ax=ax1, label='Inclination (degrees)')
        
        ax1.set_xlabel('Semi-major Axis (AU)')
        ax1.set_ylabel('Eccentricity')
        ax1.set_title('Semi-major Axis vs. Eccentricity')
        
        # Add line for parabolic orbits (e=1)
        ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        ax1.text(ax1.get_xlim()[1] * 0.95, 1.02, 'Parabolic', color='r')
    
    # Plot 2: Perihelion Distance (q) vs. Inclination (incl)
    if '_q' in df.columns and '_incl' in df.columns:
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Filter out extreme values
        plot_data = df[(df['_q'] > 0) & (df['_q'] < 5) & (df['_incl'] >= 0) & (df['_incl'] <= 180)]
        
        scatter = ax2.scatter(
            plot_data['_q'], 
            plot_data['_incl'], 
            c=plot_data.get('_e', 'blue'),
            cmap='plasma',
            alpha=0.5,
            s=5
        )
        
        if '_e' in df.columns:
            plt.colorbar(scatter, ax=ax2, label='Eccentricity')
        
        ax2.set_xlabel('Perihelion Distance (AU)')
        ax2.set_ylabel('Inclination (degrees)')
        ax2.set_title('Perihelion Distance vs. Inclination')
        
        # Add line for Earth's orbit (q ≈ 1 AU)
        ax2.axvline(x=1, color='g', linestyle='--', alpha=0.7)
        ax2.text(1.02, ax2.get_ylim()[1] * 0.95, 'Earth', color='g')
    
    # Plot 3: Histogram of Semi-major Axis (a)
    if '_a' in df.columns:
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Filter out extreme values
        plot_data = df[(df['_a'] > 0) & (df['_a'] < 100)]
        
        sns.histplot(plot_data['_a'], bins=50, kde=True, ax=ax3)
        ax3.set_xlabel('Semi-major Axis (AU)')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Semi-major Axis')
        
        # Add vertical line for Jupiter (a ≈ 5.2 AU)
        ax3.axvline(x=5.2, color='orange', linestyle='--', alpha=0.7)
        ax3.text(5.3, ax3.get_ylim()[1] * 0.95, 'Jupiter', color='orange')
    
    # Plot 4: Scatter plot of streams in a-e space (if streams available)
    ax4 = fig.add_subplot(gs[1, 1])
    
    if '_stream' in df.columns and '_a' in df.columns and '_e' in df.columns:
        # Get the top 5 streams
        top_streams = df['_stream'].value_counts().nlargest(5).index
        
        # Plot each stream in a different color
        for stream in top_streams:
            stream_data = df[df['_stream'] == stream]
            stream_data = stream_data[(stream_data['_a'] > 0) & (stream_data['_a'] < 100) & 
                                      (stream_data['_e'] >= 0) & (stream_data['_e'] <= 1.5)]
            
            ax4.scatter(
                stream_data['_a'], 
                stream_data['_e'], 
                label=stream,
                alpha=0.7,
                s=10
            )
        
        ax4.legend(title='Meteor Stream')
        ax4.set_xlabel('Semi-major Axis (AU)')
        ax4.set_ylabel('Eccentricity')
        ax4.set_title('Orbital Elements by Stream')
        
        # Add line for parabolic orbits (e=1)
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    else:
        # If stream information is not available, create a different plot
        if '_peri' in df.columns and '_node' in df.columns:
            plot_data = df[(df['_peri'] >= 0) & (df['_peri'] <= 360) & 
                          (df['_node'] >= 0) & (df['_node'] <= 360)]
            
            ax4.scatter(
                plot_data['_node'], 
                plot_data['_peri'], 
                alpha=0.5,
                s=5
            )
            ax4.set_xlabel('Longitude of Ascending Node (degrees)')
            ax4.set_ylabel('Argument of Perihelion (degrees)')
            ax4.set_title('Orbital Orientation')
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, "Not enough orbital element data", 
                    horizontalalignment='center', verticalalignment='center')
    
    # Add overall title and adjust layout
    plt.suptitle('Meteor Orbital Elements Analysis', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    if save_path is None:
        output_dir = create_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f'orbital_elements_{timestamp}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved orbital elements plot to {save_path}")
    return save_path


def plot_stream_analysis(df: pd.DataFrame, save_path: Optional[str] = None) -> str:
    """
    Plot detailed analysis of meteor streams.
    
    Args:
        df: Input DataFrame
        save_path: Path to save the plot (if None, use default location)
        
    Returns:
        Path to the saved plot
    """
    if '_stream' not in df.columns:
        logger.warning("Required column '_stream' not found")
        return None
    
    # Get the top 10 streams by count
    stream_counts = df['_stream'].value_counts()
    top_streams = stream_counts.nlargest(10).index
    
    # Filter data to include only top streams
    stream_data = df[df['_stream'].isin(top_streams)]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Bar chart of meteor counts by stream
    ax1 = axs[0, 0]
    stream_counts[top_streams].plot(kind='bar', ax=ax1)
    ax1.set_xlabel('Meteor Stream')
    ax1.set_ylabel('Count')
    ax1.set_title('Number of Meteors by Stream')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Box plot of magnitudes by stream
    ax2 = axs[0, 1]
    if '_amag' in df.columns:
        sns.boxplot(x='_stream', y='_amag', data=stream_data, ax=ax2)
        ax2.set_xlabel('Meteor Stream')
        ax2.set_ylabel('Apparent Magnitude')
        ax2.set_title('Magnitude Distribution by Stream')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, "Magnitude data not available", 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot 3: Average orbital elements by stream (bar chart)
    ax3 = axs[1, 0]

    orbital_elements = ['_a', '_e', '_incl', '_q']
    available_elements = [col for col in orbital_elements if col in df.columns]

    if len(available_elements) >= 3:
        # Calculate mean values for each stream and element
        means = df.groupby('_stream')[available_elements].mean()
        means = means.loc[top_streams[:5]]  # Use only top 5 for clarity

        # Normalize the data for comparison
        normalized_means = means.copy()
        for col in normalized_means.columns:
            col_min = normalized_means[col].min()
            col_max = normalized_means[col].max()
            if col_max - col_min > 0:
                normalized_means[col] = (normalized_means[col] - col_min) / (col_max - col_min)
            else:
                normalized_means[col] = 0

        # Create grouped bar chart
        x = np.arange(len(normalized_means.index))
        width = 0.2

        for i, col in enumerate(available_elements):
            offset = (i - len(available_elements)/2) * width
            ax3.bar(x + offset, normalized_means[col], width, label=col)

        ax3.set_xlabel('Meteor Stream')
        ax3.set_ylabel('Normalized Value')
        ax3.set_title('Normalized Orbital Elements by Stream')
        ax3.set_xticks(x)
        ax3.set_xticklabels(normalized_means.index, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, "Not enough orbital element data",
                horizontalalignment='center', verticalalignment='center')
    
    # Plot 4: Scatter plot of stream activity over time (if date/time info available)
    ax4 = axs[1, 1]
    
    time_cols = ['_mjd', '_sol', '_localtime']
    time_col = next((col for col in time_cols if col in df.columns), None)
    
    if time_col and time_col == '_localtime':
        try:
            # Convert to datetime if it's a string
            df['_datetime'] = pd.to_datetime(df['_localtime'])
            time_col = '_datetime'
        except:
            time_col = None
    
    if time_col:
        # Create a scatter plot of stream activity over time
        for stream in top_streams[:5]:  # Use only top 5 for clarity
            stream_subset = df[df['_stream'] == stream]
            ax4.scatter(
                stream_subset[time_col], 
                np.ones(len(stream_subset)) * list(top_streams[:5]).index(stream),
                label=stream,
                alpha=0.5,
                s=10
            )
        
        ax4.set_xlabel('Time')
        ax4.set_yticks(range(len(top_streams[:5])))
        ax4.set_yticklabels(top_streams[:5])
        ax4.set_title('Stream Activity Over Time')
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, "Time data not available", 
                horizontalalignment='center', verticalalignment='center')
    
    # Add overall title and adjust layout
    plt.suptitle('Meteor Stream Analysis', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    if save_path is None:
        output_dir = create_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f'stream_analysis_{timestamp}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved stream analysis plot to {save_path}")
    return save_path


def create_visualization_report(df: pd.DataFrame, output_dir: Optional[str] = None) -> str:
    """
    Create a comprehensive visualization report with multiple plots.
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save the plots (if None, use default location)
        
    Returns:
        Path to the report directory
    """
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = create_output_dir(f'report_{timestamp}')
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_paths = []
    
    # Plot 1: Meteor Distribution
    distribution_path = plot_meteor_distribution(
        df, 
        save_path=os.path.join(output_dir, 'meteor_distribution.png')
    )
    if distribution_path:
        plot_paths.append(distribution_path)
    
    # Plot 2: Velocity Distribution
    velocity_path = plot_velocity_distribution(
        df, 
        save_path=os.path.join(output_dir, 'velocity_distribution.png')
    )
    if velocity_path:
        plot_paths.append(velocity_path)
    
    # Plot 3: Orbital Elements
    orbital_path = plot_orbital_elements(
        df, 
        save_path=os.path.join(output_dir, 'orbital_elements.png')
    )
    if orbital_path:
        plot_paths.append(orbital_path)
    
    # Plot 4: Stream Analysis
    stream_path = plot_stream_analysis(
        df, 
        save_path=os.path.join(output_dir, 'stream_analysis.png')
    )
    if stream_path:
        plot_paths.append(stream_path)
    
    # Create an HTML report
    html_path = os.path.join(output_dir, 'report.html')
    
    with open(html_path, 'w') as f:
        f.write('<html>\n<head>\n')
        f.write('<title>Meteor Data Visualization Report</title>\n')
        f.write('<style>\n')
        f.write('body { font-family: Arial, sans-serif; margin: 20px; }\n')
        f.write('h1 { color: #336699; }\n')
        f.write('img { max-width: 100%; height: auto; margin: 20px 0; }\n')
        f.write('</style>\n')
        f.write('</head>\n<body>\n')
        
        f.write('<h1>Meteor Data Visualization Report</h1>\n')
        f.write(f'<p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
        
        for plot_path in plot_paths:
            plot_name = os.path.basename(plot_path)
            plot_title = ' '.join(plot_name.split('_')[:-1]).title()
            
            f.write(f'<h2>{plot_title}</h2>\n')
            f.write(f'<img src="{plot_name}" alt="{plot_title}">\n')
        
        f.write('</body>\n</html>')
    
    logger.info(f"Generated visualization report at {html_path}")
    return output_dir


def plot_velocity_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> str:
    """
    Plot the distribution of meteor velocities.
    
    Args:
        df: Input DataFrame
        save_path: Path to save the plot (if None, use default location)
        
    Returns:
        Path to the saved plot
    """
    if '_vg' not in df.columns:
        logger.warning("Required column '_vg' not found")
        return None
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    sns.histplot(df['_vg'].dropna(), bins=50, kde=True, ax=ax1)
    ax1.set_xlabel('Geocentric Velocity (km/s)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Meteor Velocities')
    
    # Box plot by stream (if available)
    if '_stream' in df.columns:
        # Get the top 10 streams by count
        top_streams = df['_stream'].value_counts().nlargest(10).index
        
        # Filter data to include only top streams
        stream_data = df[df['_stream'].isin(top_streams)]
        
        # Create box plot
        sns.boxplot(x='_stream', y='_vg', data=stream_data, ax=ax2)
        ax2.set_xlabel('Meteor Stream')
        ax2.set_ylabel('Geocentric Velocity (km/s)')
        ax2.set_title('Velocity Distribution by Stream')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    else:
        # If stream information is not available, create a different plot
        if '_vo' in df.columns and '_vi' in df.columns:
            ax2.scatter(df['_vi'], df['_vo'], alpha=0.5, s=5)
            ax2.set_xlabel('Initial Velocity (km/s)')
            ax2.set_ylabel('Observed Velocity (km/s)')
            ax2.set_title('Initial vs Observed Velocities')
        else:
            ax2.axis('off')
            ax2.text(0.5, 0.5, "Stream data not available",
                    horizontalalignment='center', verticalalignment='center')

    # Save the plot
    if save_path is None:
        output_dir = create_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f'velocity_distribution_{timestamp}.png')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved velocity distribution plot to {save_path}")
    return save_path