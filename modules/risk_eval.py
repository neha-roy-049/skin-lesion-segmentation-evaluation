def compute_risk_score(features, confidence_score=1.0):
    A = features['asymmetry']
    B = features['border']
    C = features['color']
    D = features['diameter']

    raw_score = (1.3 * A) + (0.1 * B) + (0.5 * C) + (0.5 * D)
    max_possible = 2.4

    normalized = (raw_score / max_possible) * 100
    adjusted = normalized * confidence_score + 50 * (1 - confidence_score)

    if adjusted < 35:
        risk_class = "Low Risk"
        color = "green"
    elif adjusted < 65:
        risk_class = "Medium Risk"
        color = "orange"
    else:
        risk_class = "High Risk"
        color = "red"

    return {
        'raw_score': round(raw_score, 3),
        'normalized_score': round(normalized, 1),
        'adjusted_score': round(adjusted, 1),
        'risk_class': risk_class,
        'color': color,
        'confidence': round(confidence_score * 100, 1)
    }