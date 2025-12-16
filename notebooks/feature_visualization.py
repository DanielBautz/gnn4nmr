import matplotlib.pyplot as plt
import numpy as np

def get_feature_names(node_type):
    """Map feature indices to readable German names."""
    if node_type == 'H':
        names = (['H', 'Li', 'B', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl'] +
                ['Masse', 'Formalladung', 'Grad', 'NMR-Shift', 'CN(X)',
                 'Shift_NC', 'Dia_Abschirm', 'Para_Abschirm', 'Span', 'Skew',
                 'Asymmetrie', 'Anisotropie', 'Mull_Ladung', 'Loew_Ladung',
                 'Mull_s', 'Mull_p', 'Loew_s', 'Loew_p', 'BO_Loew', 'BO_Mayer', 'Mayer_VA'])
    elif node_type == 'C':
        names = (['H', 'Li', 'B', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl'] +
                ['Masse', 'Formalladung', 'Grad', 'NMR-Shift', 'CN(X)',
                 'Dia_Abschirm', 'Para_Abschirm', 'Span', 'Skew', 'Asymmetrie', 'Anisotropie',
                 'Mull_Ladung', 'Loew_Ladung', 'Mull_s', 'Mull_p', 'Mull_d', 'Mull_p_std',
                 'Loew_s', 'Loew_p', 'Loew_d', 'Loew_p_std', 'BO_Loew_Sum', 'BO_Loew_Avg',
                 'BO_Mayer_Sum', 'BO_Mayer_Avg', 'Mayer_VA'])
    else:
        names = ['H', 'Li', 'B', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl'] + ['Masse', 'Formalladung', 'Grad']

    return names

def plot_feature_importance(batch_results, show_all=True):
    """Create bar charts showing feature importance with German names - both absolute and signed."""
    import matplotlib.pyplot as plt

    # Color mapping for different node types
    colors = {'H': 'blue', 'C': 'green', 'Others': 'red'}

    for node_type, result in batch_results.items():
        if not result.get('avg_importance'):
            continue

        # Check explainer type to determine which plots to show
        explainer_type = result.get('explainer_type', 'gnn')  # Default to gnn for backward compatibility

        # Get feature names
        feature_names = get_feature_names(node_type)

        # ===== PLOT 1: Absolute Importance (for ranking) - Only for IG explainer =====
        if explainer_type == 'ig':
            if result.get('avg_abs_importance'):
                abs_importance_values = np.array(result['avg_abs_importance'])
                abs_importance_type = "absolute"
            else:
                abs_importance_values = np.abs(np.array(result['avg_importance']))
                abs_importance_type = "absolute (from raw)"

            # Sort all features by absolute importance (descending)
            feature_indices = np.arange(len(abs_importance_values))
            sorted_indices_abs = np.argsort(abs_importance_values)[::-1]  # Descending order
            sorted_values_abs = abs_importance_values[sorted_indices_abs]
            sorted_names_abs = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in sorted_indices_abs]

            # Create figure for absolute importance
            if show_all:
                fig_height = max(12, len(sorted_names_abs) * 0.3)
                plt.figure(figsize=(16, fig_height))
            else:
                plt.figure(figsize=(12, 8))

            # Create horizontal bar chart for absolute importance
            bars = plt.barh(range(len(sorted_names_abs)), sorted_values_abs,
                           color=colors.get(node_type, 'gray'), alpha=0.7)

            plt.yticks(range(len(sorted_names_abs)), sorted_names_abs, fontsize=8)
            plt.xlabel('Absolute durchschnittliche Wichtigkeit', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title(f'Absolute Feature-Wichtigkeit für {node_type}-Atome ({abs_importance_type})', fontsize=14)
            plt.grid(axis='x', alpha=0.3)

            # Add value labels on bars
            max_value = max(sorted_values_abs)
            for i, (bar, value) in enumerate(zip(bars, sorted_values_abs)):
                if value > max_value * 0.01:
                    plt.text(bar.get_width() + max_value * 0.005,
                            bar.get_y() + bar.get_height()/2,
                            f'{value:.3f}', ha='left', va='center', fontsize=6)

            plt.tight_layout()
            plt.show()

        # ===== PLOT 2: Signed Importance (for direction) =====
        signed_importance_values = np.array(result['avg_importance'])

        # Sort by absolute value for ranking, but show signed values
        sorted_indices_signed = np.argsort(np.abs(signed_importance_values))[::-1]  # Sort by abs, descending
        sorted_values_signed = signed_importance_values[sorted_indices_signed]
        sorted_names_signed = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in sorted_indices_signed]

        # Create figure for signed importance
        if show_all:
            fig_height = max(12, len(sorted_names_signed) * 0.3)
            plt.figure(figsize=(16, fig_height))
        else:
            plt.figure(figsize=(12, 8))

        # Color bars based on sign (positive = green, negative = red) - only for IG
        if explainer_type == 'ig':
            bar_colors = ['green' if val >= 0 else 'red' for val in sorted_values_signed]
            title_suffix = 'mit Richtung'
        else:  # gnn
            bar_colors = colors.get(node_type, 'gray')
            title_suffix = ''

        # Create horizontal bar chart for signed importance
        bars = plt.barh(range(len(sorted_names_signed)), sorted_values_signed,
                       color=bar_colors, alpha=0.7)

        plt.yticks(range(len(sorted_names_signed)), sorted_names_signed, fontsize=8)
        plt.xlabel('Durchschnittliche Wichtigkeit', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Feature-Wichtigkeit {title_suffix} für {node_type}-Atome'.strip(), fontsize=14)
        plt.grid(axis='x', alpha=0.3)

        # Add zero line only for IG (since it can have negative values)
        if explainer_type == 'ig':
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)

        # Add value labels on bars
        max_abs_value = max(np.abs(sorted_values_signed))
        for i, (bar, value) in enumerate(zip(bars, sorted_values_signed)):
            if abs(value) > max_abs_value * 0.01:
                if explainer_type == 'ig':
                    ha = 'left' if value >= 0 else 'right'
                    x_pos = bar.get_width() + (max_abs_value * 0.005 if value >= 0 else -max_abs_value * 0.005)
                else:
                    ha = 'left'
                    x_pos = bar.get_width() + max_abs_value * 0.005
                plt.text(x_pos, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha=ha, va='center', fontsize=6)

        plt.tight_layout()
        plt.show()

def plot_single_feature_importance(feature_importance, node_type, title_suffix=""):
    """Create bar chart showing feature importance for a single node type."""
    import matplotlib.pyplot as plt

    # Get feature names
    feature_names = get_feature_names(node_type)

    # Get importance values
    avg_importance = feature_importance

    # Sort all features by importance (descending)
    feature_indices = np.arange(len(avg_importance))
    sorted_indices = np.argsort(avg_importance)[::-1]  # Descending order
    sorted_values = avg_importance[sorted_indices]
    sorted_names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in sorted_indices]

    # Create figure with appropriate size
    fig_height = max(12, len(sorted_names) * 0.3)  # Scale height based on number of features
    plt.figure(figsize=(16, fig_height))

    # Create horizontal bar chart
    bars = plt.barh(range(len(sorted_names)), sorted_values,
                   color='blue', alpha=0.7)

    plt.yticks(range(len(sorted_names)), sorted_names, fontsize=8)
    plt.xlabel('Wichtigkeit', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Alle Features nach Wichtigkeit für {node_type}-Atom{title_suffix}', fontsize=14)
    plt.grid(axis='x', alpha=0.3)

    # Add value labels on bars (only for significant values)
    max_value = max(sorted_values)
    for i, (bar, value) in enumerate(zip(bars, sorted_values)):
        if abs(value) > max_value * 0.01:  # Only show labels for values > 1% of max
            plt.text(bar.get_width() + max_value * 0.005,
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontsize=6)

    plt.tight_layout()
    plt.show()
