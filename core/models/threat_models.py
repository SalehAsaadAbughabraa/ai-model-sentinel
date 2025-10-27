# Threat models for threat analytics

class ThreatIndicator:
    def __init__(self, indicator_type, value, severity):
        self.indicator_type = indicator_type
        self.value = value
        self.severity = severity

class ThreatAssessment:
    def __init__(self):
        self.indicators = []
    
    def add_indicator(self, indicator):
        self.indicators.append(indicator)

class ThreatAnalyticsEngine:
    def __init__(self):
        self.name = "ThreatAnalyticsEngine"

ThreatAnalyticsEngine = ThreatAnalyticsEngine
