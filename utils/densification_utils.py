import torch

def extract_candidate_positions(beams: torch.Tensor, importance_scores: torch.Tensor, min_dist: float, device: torch.device) -> torch.Tensor:
    """
    Extracts optimal locations for new Gaussians by computing geometric intersections 
    of anomalous beams and applying Continuous Spatial NMS.

    Args:
        beams: Tensor of shape (N, 2, 2) representing the start and end points of the beams.
        importance_scores: Tensor of shape (N,) representing the metric to evaluate (measurements or residuals).
        min_dist: Minimum spatial distance between generated points to avoid redundancy.
        device: Torch device.
        
    Returns:
        Tensor of shape (M, 2) with the filtered candidate positions.
    """
    p0 = beams[:, 0, :]
    p1 = beams[:, 1, :]
    lengths = torch.norm(p1 - p0, dim=1)
    
    normalized_scores = importance_scores / torch.clamp(lengths, min=1e-5)
    
    pos_mask = normalized_scores > 0
    if not pos_mask.any():
        return torch.empty((0, 2), device=device)
        
    # 1. Statistical Filter: Mean + Std
    valid_scores = normalized_scores[pos_mask]
    mean_score = torch.mean(valid_scores)
    std_score = torch.std(valid_scores)
    
    threshold = mean_score + 1.0 * (std_score + 1e-6)
    
    candidate_mask = pos_mask & (normalized_scores > threshold)
    candidate_indices = torch.where(candidate_mask)[0]
    
    num_candidates = candidate_indices.shape[0]
    if num_candidates < 2:
        return torch.empty((0, 2), device=device)

    # 2. Geometric Intersections of candidate segments
    candidate_beams = beams[candidate_indices]
    candidate_scores = normalized_scores[candidate_indices]
    
    i, j = torch.triu_indices(num_candidates, num_candidates, offset=1, device=device)
    
    b_i = candidate_beams[i] 
    b_j = candidate_beams[j] 
    
    x1, y1 = b_i[:, 0, 0], b_i[:, 0, 1]
    x2, y2 = b_i[:, 1, 0], b_i[:, 1, 1]
    x3, y3 = b_j[:, 0, 0], b_j[:, 0, 1]
    x4, y4 = b_j[:, 1, 0], b_j[:, 1, 1]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    valid_intersection = torch.abs(denom) > 1e-7
    
    # Parametric line equations (t for beam_i, u for beam_j)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / (denom + 1e-9)
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / (denom + 1e-9)
    
    # A true segment intersection ONLY occurs if both t and u are between 0 and 1.
    # We use 0.01 and 0.99 to avoid intersections exactly at the emitters (radial origins).
    in_bounds = (t >= 0.01) & (t <= 0.99) & (u >= 0.01) & (u <= 0.99)
    keep_mask = valid_intersection & in_bounds
    
    if not keep_mask.any():
        return torch.empty((0, 2), device=device)
        
    # Calculate exact coordinates for valid intersections
    px = x1[keep_mask] + t[keep_mask] * (x2[keep_mask] - x1[keep_mask])
    py = y1[keep_mask] + t[keep_mask] * (y2[keep_mask] - y1[keep_mask])
    
    intersections = torch.stack([px, py], dim=-1)
    int_scores = candidate_scores[i[keep_mask]] + candidate_scores[j[keep_mask]]

    # 3. Continuous Spatial Non-Maximum Suppression (NMS)
    sorted_idx = torch.argsort(int_scores, descending=True)
    sorted_points = intersections[sorted_idx]
    
    keep_points = []
    for pt in sorted_points:
        if len(keep_points) > 0:
            kept_tensor = torch.stack(keep_points)
            distances = torch.norm(kept_tensor - pt, dim=1)
            if torch.min(distances) < min_dist:
                continue 
        keep_points.append(pt)
            
    return torch.stack(keep_points)