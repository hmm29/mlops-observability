def compare_model_versions(registry_client, model_name, version1, version2, metric="accuracy"):
    """
    Compare two model versions based on a specific metric
    
    Args:
        registry_client: ModelRegistry client instance
        model_name: Name of the model
        version1: First version to compare
        version2: Second version to compare
        metric: Metric name to compare
        
    Returns:
        Dictionary with comparison results
    """
    run1 = registry_client.client.get_run(
        registry_client.client.get_model_version(model_name, version1).run_id
    )
    run2 = registry_client.client.get_run(
        registry_client.client.get_model_version(model_name, version2).run_id
    )
    
    metric1 = run1.data.metrics.get(metric, 0)
    metric2 = run2.data.metrics.get(metric, 0)
    
    return {
        "version1": version1,
        "version2": version2,
        "metric": metric,
        "difference": metric1 - metric2,
        "percentage_change": (metric1 - metric2) / metric2 * 100 if metric2 != 0 else float('inf')
    }

def find_best_model_version(registry_client, model_name, metric="accuracy", max_results=5):
    """
    Find the best performing model version based on a metric
    
    Args:
        registry_client: ModelRegistry client instance
        model_name: Name of the model
        metric: Metric to optimize (default: accuracy)
        max_results: Maximum number of versions to consider
        
    Returns:
        Best model version and its metric value
    """
    versions = registry_client.get_model_versions(model_name)
    
    # Only consider the most recent versions
    versions = versions[:max_results] if len(versions) > max_results else versions
    
    best_version = None
    best_metric = float('-inf')
    
    for version in versions:
        run = registry_client.client.get_run(version.run_id)
        if metric in run.data.metrics:
            metric_value = run.data.metrics[metric]
            if metric_value > best_metric:
                best_metric = metric_value
                best_version = version.version
    
    return {
        "version": best_version,
        "metric": metric,
        "value": best_metric
    }
