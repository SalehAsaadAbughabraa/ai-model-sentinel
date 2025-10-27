# architecture_diagram.py
import json
from pathlib import Path

class ArchitectureDiagram:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "enterprise_sentinel_docs_v2"
    
    def load_engine_data(self):
        engines = []
        engines_dir = self.docs_dir / "engines"
        
        for md_file in engines_dir.glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            
            category = "Other"
            if "**Category:**" in content:
                category_line = [line for line in content.split('\n') if "**Category:**" in line][0]
                category = category_line.split("**Category:**")[1].strip()
            
            engines.append({
                "name": md_file.stem,
                "category": category,
                "file": md_file.name
            })
        
        return engines
    
    def generate_architecture_diagram(self):
        print("Generating architecture diagram...")
        
        engines = self.load_engine_data()
        
        mermaid_content = """```mermaid
graph TB
    classDef quantum fill:#e1f5fe
    classDef security fill:#fce4ec
    classDef ml fill:#e8f5e8
    classDef data fill:#fff3e0
    classDef other fill:#f3e5f5
    
    subgraph "AI Sentinel Architecture"
"""
        
        categories = {}
        for engine in engines:
            cat = engine["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(engine["name"])
        
        for category, engine_list in categories.items():
            mermaid_content += f'\n        subgraph "{category} Engines"\n'
            for engine in engine_list:
                mermaid_content += f'            {engine.replace(" ", "_")}[{engine}]\n'
            mermaid_content += '        end\n'
        
        mermaid_content += """    
    end

    %% Connections
    quantum_engines_fixed --> enterprise_security
    anomaly_engine --> drift_detector
    dynamic_engine_fixed --> quantum_engines_fixed
"""
        
        mermaid_content += "\n```"
        
        diagram_file = self.docs_dir / "diagrams" / "architecture.mmd"
        diagram_file.write_text(mermaid_content, encoding='utf-8')
        
        print(f"Architecture diagram generated: {diagram_file}")
        return mermaid_content
    
    def create_system_overview(self):
        print("Creating system overview document...")
        
        analysis_file = self.docs_dir / "project_analysis.json"
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        overview_content = """# System Architecture Overview

## High-Level Design

Enterprise AI Sentinel v2.0 follows a modular architecture with specialized engines.

## Component Categories

### Quantum Engines
- Advanced quantum computing integration
- Cryptographic and mathematical operations
- Quantum-enhanced algorithms

### Security Engines
- Threat detection and analysis
- Security monitoring
- Audit and compliance

### AI/ML Engines
- Machine learning model management
- Training and inference pipelines
- Model monitoring and validation

### Data Engines
- Data processing and analytics
- Database operations
- Big data integration

### Monitoring & Fusion Engines
- System performance monitoring
- Engine fusion and coordination
- Real-time analytics

## Data Flow
1. **Input** -> Security Validation -> Data Processing
2. **Processing** -> AI/ML Analysis -> Quantum Enhancement
3. **Output** -> Monitoring -> Storage/APIs

## Integration Points
- Internal engine communication via message bus
- External APIs for client applications
- Database connections for persistence
- Monitoring systems for observability
"""

        overview_file = self.docs_dir / "architecture_overview.md"
        overview_file.write_text(overview_content, encoding='utf-8')
        
        print(f"System overview created: {overview_file}")

if __name__ == "__main__":
    diagram = ArchitectureDiagram(".")
    diagram.generate_architecture_diagram()
    diagram.create_system_overview()
    print("Architecture documentation completed!")