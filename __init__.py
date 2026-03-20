from .data_loader import load_data, subject_split, get_subject_windows
from .preprocessing import drop_missing, scale_channels, clip_artefacts
from .feature_extraction import extract_subject_features, extract_window_features, build_feature_names
from .evaluation import evaluate, plot_confusion_matrix, plot_roc_curve, compare_models, aggregate_by_subject
from .baseline import train_baselines, CLASSIFIERS
from .neural_net import ADHDNet, train_nn, EEGConvNet, train_cnn, prepare_cnn_input

__all__ = [
    'load_data', 'subject_split', 'get_subject_windows',
    'drop_missing', 'scale_channels', 'clip_artefacts',
    'extract_subject_features', 'extract_window_features', 'build_feature_names',
    'evaluate', 'plot_confusion_matrix', 'plot_roc_curve', 'compare_models', 'aggregate_by_subject',
    'train_baselines', 'CLASSIFIERS',
    'ADHDNet', 'train_nn', 'EEGConvNet', 'train_cnn', 'prepare_cnn_input',
]
