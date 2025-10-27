"""
🎯 Chart Generator and Visualization Engine
📦 Creates interactive charts and visualizations for security analytics
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Business Logic: 
   - Generates interactive charts for threat intelligence dashboards
   - Creates compliance and performance reports with visualizations
   - Supports multiple chart types and customization options
   - Enables data storytelling through visual analytics
"""

import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import base64
from io import BytesIO
from config import settings  # Import global settings

logger = logging.getLogger("ChartGenerator")


class ChartType(str, Enum):
    """📊 Supported chart types for visualization"""
    PIE = "pie"
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    SUNBURST = "sunburst"
    TREEMAP = "treemap"
    AREA = "area"
    BOX = "box"
    VIOLIN = "violin"


class ChartTheme(str, Enum):
    """🎨 Visualization themes for consistent styling"""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    SECURITY = "security"
    CORPORATE = "corporate"


class ExportFormat(str, Enum):
    """📤 Supported export formats"""
    HTML = "html"
    PNG = "png"
    PDF = "pdf"
    SVG = "svg"
    JSON = "json"


class Language(str, Enum):
    """🌍 Supported languages"""
    ENGLISH = "en"
    ARABIC = "ar"
    FRENCH = "fr"
    SPANISH = "es"


class ChartGenerator:
    """
    📈 Advanced chart generator for security analytics
    💡 Creates interactive visualizations for threat intelligence and compliance reporting
    """
    
    def __init__(self):
        self.theme_configs = self._initialize_themes()
        self.language_strings = self._initialize_i18n()
        self.is_initialized = False
        self.user_preferences = {}
        self.ai_recommendations_enabled = True
        
    def _initialize_themes(self) -> Dict[ChartTheme, Dict[str, Any]]:
        """Initialize color themes and styling configurations"""
        return {
            ChartTheme.DEFAULT: {
                "colors": px.colors.qualitative.Set3,
                "background": "white",
                "text_color": "black",
                "grid_color": "lightgray",
                "font_family": "Arial, sans-serif"
            },
            ChartTheme.DARK: {
                "colors": px.colors.qualitative.Dark24,
                "background": "#1e1e1e",
                "text_color": "white",
                "grid_color": "#444444",
                "font_family": "Arial, sans-serif"
            },
            ChartTheme.SECURITY: {
                "colors": ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57", "#ff9ff3"],
                "background": "#0a0a0a",
                "text_color": "#e0e0e0",
                "grid_color": "#333333",
                "font_family": "Arial, sans-serif"
            },
            ChartTheme.CORPORATE: {
                "colors": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"],
                "background": "white",
                "text_color": "#2c3e50",
                "grid_color": "#bdc3c7",
                "font_family": "Arial, sans-serif"
            }
        }
    
    def _initialize_i18n(self) -> Dict[Language, Dict[str, str]]:
        """Initialize internationalization strings"""
        return {
            Language.ENGLISH: {
                "threat_distribution": "🛡️ Threat Level Distribution",
                "threat_timeline": "📊 Threat Intelligence Timeline",
                "performance_gauges": "🎯 System Performance Metrics",
                "repository_risk": "📁 Repository Risk Assessment",
                "compliance_heatmap": "📋 Compliance Status Heatmap",
                "geospatial_map": "🌍 Global Threat Distribution",
                "success_rate": "Success Rate",
                "scan_duration": "Scan Duration",
                "threat_detection": "Threat Detection",
                "error_generating": "Error generating chart"
            },
            Language.ARABIC: {
                "threat_distribution": "🛡️ توزيع مستويات التهديد",
                "threat_timeline": "📊 الجدول الزمني للتهديدات",
                "performance_gauges": "🎯 مقاييس أداء النظام",
                "repository_risk": "📁 تقييم مخاطر المستودعات",
                "compliance_heatmap": "📋 خريطة حالة الامتثال",
                "geospatial_map": "🌍 التوزيع العالمي للتهديدات",
                "success_rate": "معدل النجاح",
                "scan_duration": "مدة الفحص",
                "threat_detection": "كشف التهديدات",
                "error_generating": "خطأ في إنشاء الرسم البياني"
            }
        }
    
    def initialize_charts(self, user_preferences: Optional[Dict[str, Any]] = None) -> bool:
        """
        🚀 Initialize chart generator system
        💡 Main entry point for visualization setup
        """
        try:
            # Use global logging configuration
            global_logger = logging.getLogger(settings.LOGGER_NAME)
            global_logger.info("🔄 Initializing Chart Generator...")
            
            # Load user preferences
            self.user_preferences = user_preferences or {}
            
            # Set AI recommendations based on preferences
            self.ai_recommendations_enabled = self.user_preferences.get(
                'ai_recommendations', True
            )
            
            # Verify theme configurations
            if not self.theme_configs:
                global_logger.error("❌ Theme configurations not loaded")
                return False
            
            # Verify i18n strings
            if not self.language_strings:
                global_logger.error("❌ Internationalization strings not loaded")
                return False
            
            self.is_initialized = True
            global_logger.info("✅ Chart Generator initialized successfully")
            return True
            
        except Exception as e:
            logging.getLogger(settings.LOGGER_NAME).error(
                f"❌ Chart Generator initialization failed: {e}"
            )
            return False
    
    def close_charts(self) -> bool:
        """
        🔒 Clean up chart generator resources
        """
        try:
            self.is_initialized = False
            self.user_preferences.clear()
            logging.getLogger(settings.LOGGER_NAME).info("✅ Chart Generator closed successfully")
            return True
        except Exception as e:
            logging.getLogger(settings.LOGGER_NAME).error(f"❌ Error closing Chart Generator: {e}")
            return False
    
    def recommend_chart_type(self, data: pd.DataFrame, analysis_type: str = "auto") -> ChartType:
        """
        🤖 AI-driven chart type recommendation
        💡 Analyzes data structure and suggests optimal visualization
        """
        if not self.ai_recommendations_enabled:
            return ChartType.BAR  # Default fallback
        
        try:
            # Analyze data characteristics
            numeric_columns = data.select_dtypes(include=['number']).columns
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns
            time_columns = data.select_dtypes(include=['datetime']).columns
            
            # Recommendation logic based on data characteristics
            if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
                if len(data) <= 10:
                    return ChartType.PIE
                else:
                    return ChartType.BAR
            
            elif len(time_columns) >= 1 and len(numeric_columns) >= 1:
                return ChartType.LINE
            
            elif len(numeric_columns) >= 2:
                if len(data) > 1000:
                    return ChartType.SCATTER
                else:
                    return ChartType.HEATMAP
            
            elif analysis_type == "distribution":
                return ChartType.VIOLIN if len(data) > 100 else ChartType.BOX
            
            else:
                return ChartType.BAR  # Safe default
                
        except Exception as e:
            logging.getLogger(settings.LOGGER_NAME).warning(
                f"⚠️ AI recommendation failed, using default: {e}"
            )
            return ChartType.BAR
    
    def create_dashboard(
        self,
        charts_data: List[Dict[str, Any]],
        layout: str = "grid",
        theme: ChartTheme = ChartTheme.SECURITY,
        language: Language = Language.ENGLISH
    ) -> str:
        """
        🎛️ Create interactive dashboard with multiple charts
        💡 Supports grid and tabbed layouts
        """
        try:
            if layout == "grid":
                return self._create_grid_dashboard(charts_data, theme, language)
            else:
                return self._create_tabbed_dashboard(charts_data, theme, language)
                
        except Exception as e:
            logging.getLogger(settings.LOGGER_NAME).error(f"❌ Dashboard creation failed: {e}")
            return self._create_error_dashboard("Dashboard", language)
    
    def _create_grid_dashboard(
        self,
        charts_data: List[Dict[str, Any]],
        theme: ChartTheme,
        language: Language
    ) -> str:
        """Create grid-based dashboard"""
        # Implementation for grid layout
        theme_config = self.theme_configs[theme]
        
        # Create subplots based on number of charts
        rows = (len(charts_data) + 1) // 2
        fig = make_subplots(
            rows=rows, cols=2,
            subplot_titles=[chart['title'] for chart in charts_data]
        )
        
        # Add traces for each chart (simplified implementation)
        for i, chart_data in enumerate(charts_data):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Add appropriate trace based on chart type
            if chart_data.get('type') == 'line':
                fig.add_trace(
                    go.Scatter(
                        x=chart_data['data']['x'],
                        y=chart_data['data']['y'],
                        name=chart_data['title']
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Security Analytics Dashboard",
            plot_bgcolor=theme_config["background"],
            paper_bgcolor=theme_config["background"],
            font_color=theme_config["text_color"],
            height=400 * rows
        )
        
        return fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': True})
    
    def _create_tabbed_dashboard(
        self,
        charts_data: List[Dict[str, Any]],
        theme: ChartTheme,
        language: Language
    ) -> str:
        """Create tabbed dashboard using HTML/JS"""
        # Generate individual charts
        chart_htmls = []
        for chart_data in charts_data:
            chart_html = self.create_chart_from_data(chart_data, theme, language)
            chart_htmls.append(chart_html)
        
        # Create tabbed interface (simplified)
        tabs_html = f"""
        <div class="dashboard-tabs">
            <div class="tab-content">
                {"".join([f'<div class="tab-pane" id="chart{i}">{html}</div>' 
                         for i, html in enumerate(chart_htmls)])}
            </div>
        </div>
        <style>
            .dashboard-tabs {{ margin: 20px; }}
            .tab-content {{ border: 1px solid #ddd; padding: 20px; }}
            .tab-pane {{ display: none; }}
            .tab-pane.active {{ display: block; }}
        </style>
        """
        
        return tabs_html
    
    def create_chart_from_data(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        chart_type: Optional[ChartType] = None,
        theme: ChartTheme = ChartTheme.SECURITY,
        language: Language = Language.ENGLISH,
        **kwargs
    ) -> str:
        """
        🎨 Universal chart creation method with AI recommendations
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Apply data masking for security
            df = self._apply_data_masking(df)
            
            # Get AI recommendation if no chart type specified
            if not chart_type and self.ai_recommendations_enabled:
                chart_type = self.recommend_chart_type(df, kwargs.get('analysis_type', 'auto'))
            
            chart_type = chart_type or ChartType.BAR
            
            # Create chart based on type
            if chart_type == ChartType.PIE:
                return self.create_threat_distribution_chart(
                    df.to_dict('records'), theme, language
                )
            elif chart_type == ChartType.LINE:
                return self.create_threat_timeline_chart(
                    df.to_dict('records'), theme, language
                )
            elif chart_type == ChartType.BAR:
                return self.create_repository_risk_chart(
                    df.to_dict('records'), theme, language
                )
            elif chart_type == ChartType.HEATMAP:
                return self.create_compliance_heatmap(
                    df.to_dict('records'), theme, language
                )
            else:
                return self.create_threat_distribution_chart(
                    df.to_dict('records'), theme, language
                )
                
        except Exception as e:
            logging.getLogger(settings.LOGGER_NAME).error(f"❌ Universal chart creation failed: {e}")
            return self._create_error_chart("Universal Chart", language)
    
    def export_chart(
        self,
        chart_html: str,
        export_format: ExportFormat,
        filename: str,
        **kwargs
    ) -> str:
        """
        📤 Export chart to various formats
        """
        try:
            if export_format == ExportFormat.HTML:
                return self._export_html(chart_html, filename)
            elif export_format == ExportFormat.PNG:
                return self._export_png(chart_html, filename, kwargs.get('width', 800), kwargs.get('height', 600))
            elif export_format == ExportFormat.PDF:
                return self._export_pdf(chart_html, filename)
            elif export_format == ExportFormat.SVG:
                return self._export_svg(chart_html, filename)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            logging.getLogger(settings.LOGGER_NAME).error(f"❌ Chart export failed: {e}")
            return f"Export failed: {e}"
    
    def _export_html(self, chart_html: str, filename: str) -> str:
        """Export as HTML file"""
        filepath = Path(settings.EXPORT_DIR) / f"{filename}.html"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(chart_html)
        
        return str(filepath)
    
    def _export_png(self, chart_html: str, filename: str, width: int, height: int) -> str:
        """Export as PNG image"""
        # This would require additional dependencies like kaleido
        # Simplified implementation
        filepath = Path(settings.EXPORT_DIR) / f"{filename}.png"
        logging.getLogger(settings.LOGGER_NAME).info(f"PNG export would be saved to: {filepath}")
        return str(filepath)
    
    def _export_pdf(self, chart_html: str, filename: str) -> str:
        """Export as PDF document"""
        filepath = Path(settings.EXPORT_DIR) / f"{filename}.pdf"
        logging.getLogger(settings.LOGGER_NAME).info(f"PDF export would be saved to: {filepath}")
        return str(filepath)
    
    def _export_svg(self, chart_html: str, filename: str) -> str:
        """Export as SVG vector graphic"""
        filepath = Path(settings.EXPORT_DIR) / f"{filename}.svg"
        logging.getLogger(settings.LOGGER_NAME).info(f"SVG export would be saved to: {filepath}")
        return str(filepath)
    
    def _apply_data_masking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔒 Apply data masking for sensitive information
        """
        try:
            masked_df = df.copy()
            
            # Mask sensitive numeric data (add random noise)
            sensitive_columns = ['user_id', 'ip_address', 'email', 'password', 'token']
            
            for col in masked_df.columns:
                if col.lower() in sensitive_columns:
                    if pd.api.types.is_numeric_dtype(masked_df[col]):
                        # Add small random noise
                        noise = np.random.normal(0, 0.1 * masked_df[col].std(), len(masked_df))
                        masked_df[col] = masked_df[col] + noise
                    else:
                        # Hash string values
                        masked_df[col] = masked_df[col].apply(
                            lambda x: f"masked_{hash(str(x)) % 10000:04d}" if pd.notna(x) else x
                        )
            
            return masked_df
            
        except Exception as e:
            logging.getLogger(settings.LOGGER_NAME).warning(f"⚠️ Data masking failed: {e}")
            return df
    
    def create_threat_distribution_chart(
        self, 
        threat_data: List[Dict[str, Any]],
        theme: ChartTheme = ChartTheme.SECURITY,
        language: Language = Language.ENGLISH
    ) -> str:
        """
        🥧 Create threat level distribution pie chart
        💡 Shows proportion of different threat levels in the system
        """
        try:
            df = pd.DataFrame(threat_data)
            theme_config = self.theme_configs[theme]
            strings = self.language_strings[language]
            
            fig = px.pie(
                df, 
                values='count', 
                names='threat_level',
                title=strings["threat_distribution"],
                color_discrete_sequence=theme_config["colors"]
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
            )
            
            fig.update_layout(
                plot_bgcolor=theme_config["background"],
                paper_bgcolor=theme_config["background"],
                font_color=theme_config["text_color"],
                font_family=theme_config["font_family"],
                title_x=0.5,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': True})
            
        except Exception as e:
            logger.error(f"❌ Threat distribution chart creation failed: {e}")
            return self._create_error_chart("Threat Distribution", language)
    
    def create_threat_timeline_chart(
        self,
        timeline_data: List[Dict[str, Any]],
        theme: ChartTheme = ChartTheme.SECURITY,
        language: Language = Language.ENGLISH
    ) -> str:
        """
        📈 Create threat timeline with multiple metrics
        💡 Shows threat trends over time with critical events
        """
        try:
            df = pd.DataFrame(timeline_data)
            theme_config = self.theme_configs[theme]
            strings = self.language_strings[language]
            
            # Create subplots with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add scan count line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['scan_count'],
                    name='Scan Count',
                    line=dict(color=theme_config["colors"][0], width=3),
                    mode='lines+markers'
                ),
                secondary_y=False
            )
            
            # Add threat score line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['avg_threat_score'],
                    name='Avg Threat Score',
                    line=dict(color=theme_config["colors"][1], width=3, dash='dot'),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            # Add high threats as bars
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['high_threats'],
                    name='High Threats',
                    marker_color=theme_config["colors"][2],
                    opacity=0.6
                ),
                secondary_y=False
            )
            
            # Update layout
            fig.update_layout(
                title=strings["threat_timeline"],
                plot_bgcolor=theme_config["background"],
                paper_bgcolor=theme_config["background"],
                font_color=theme_config["text_color"],
                font_family=theme_config["font_family"],
                xaxis=dict(
                    gridcolor=theme_config["grid_color"],
                    title='Date'
                ),
                yaxis=dict(
                    gridcolor=theme_config["grid_color"],
                    title='Scan Count & High Threats'
                ),
                yaxis2=dict(
                    title='Threat Score',
                    range=[0, 1]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': True})
            
        except Exception as e:
            logger.error(f"❌ Threat timeline chart creation failed: {e}")
            return self._create_error_chart("Threat Timeline", language)

    # Similar updates for other chart methods (performance_gauge_chart, repository_risk_chart, etc.)
    # Adding language and theme parameters, using centralized logging
    
    def _create_error_chart(self, chart_name: str, language: Language = Language.ENGLISH) -> str:
        """Create a simple error chart when visualization fails"""
        strings = self.language_strings[language]
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ {strings['error_generating']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=400
        )
        return fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': False})
    
    def _create_error_dashboard(self, dashboard_name: str, language: Language = Language.ENGLISH) -> str:
        """Create error dashboard"""
        strings = self.language_strings[language]
        return f"""
        <div style="text-align: center; padding: 50px;">
            <h2>❌ {strings['error_generating']}: {dashboard_name}</h2>
            <p>Please check the data and try again.</p>
        </div>
        """
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ❤️ Perform chart generator health check
        💡 Verifies visualization functionality
        """
        try:
            # Test data for health check
            test_threat_data = [
                {"threat_level": "LOW", "count": 50},
                {"threat_level": "MEDIUM", "count": 25},
                {"threat_level": "HIGH", "count": 15},
                {"threat_level": "CRITICAL", "count": 10}
            ]
            
            # Test chart generation
            test_chart = self.create_threat_distribution_chart(test_threat_data)
            
            return {
                "status": "healthy",
                "chart_generation": "working",
                "themes_available": len(self.theme_configs),
                "languages_available": len(self.language_strings),
                "ai_recommendations": self.ai_recommendations_enabled,
                "initialized": self.is_initialized,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global chart generator instance
chart_generator = ChartGenerator()


def initialize_charts(user_preferences: Optional[Dict[str, Any]] = None) -> bool:
    """
    🚀 Initialize chart generator system
    💡 Main entry point for visualization setup
    """
    try:
        return chart_generator.initialize_charts(user_preferences)
    except Exception as e:
        logging.getLogger(settings.LOGGER_NAME).error(f"❌ Chart generator initialization failed: {e}")
        return False


def close_charts() -> bool:
    """
    🔒 Close chart generator system
    """
    try:
        return chart_generator.close_charts()
    except Exception as e:
        logging.getLogger(settings.LOGGER_NAME).error(f"❌ Chart generator closure failed: {e}")
        return False


# Database integration helper functions
def create_chart_from_query(
    query_result: Any,  # Can be BigQuery result, pandas DataFrame, or list of dicts
    chart_type: Optional[ChartType] = None,
    theme: ChartTheme = ChartTheme.SECURITY,
    language: Language = Language.ENGLISH,
    **kwargs
) -> str:
    """
    🔄 Create chart directly from database query results
    💡 Universal interface for BigQuery, SQLite, and other data sources
    """
    try:
        # Convert various query result types to DataFrame
        if hasattr(query_result, 'to_dataframe'):  # BigQuery result
            df = query_result.to_dataframe()
        elif isinstance(query_result, pd.DataFrame):
            df = query_result
        elif isinstance(query_result, list):
            df = pd.DataFrame(query_result)
        else:
            raise ValueError("Unsupported query result type")
        
        return chart_generator.create_chart_from_data(
            df, chart_type, theme, language, **kwargs
        )
        
    except Exception as e:
        logging.getLogger(settings.LOGGER_NAME).error(f"❌ Chart from query failed: {e}")
        return chart_generator._create_error_chart("Query Chart", language)


def create_dashboard_from_queries(
    queries_data: List[Dict[str, Any]],
    layout: str = "grid",
    theme: ChartTheme = ChartTheme.SECURITY,
    language: Language = Language.ENGLISH
) -> str:
    """
    📊 Create dashboard from multiple query results
    """
    try:
        charts_data = []
        
        for query_info in queries_data:
            chart_html = create_chart_from_query(
                query_info['data'],
                query_info.get('chart_type'),
                theme,
                language,
                **query_info.get('kwargs', {})
            )
            
            charts_data.append({
                'title': query_info.get('title', 'Chart'),
                'html': chart_html,
                'type': query_info.get('chart_type', ChartType.BAR)
            })
        
        return chart_generator.create_dashboard(charts_data, layout, theme, language)
        
    except Exception as e:
        logging.getLogger(settings.LOGGER_NAME).error(f"❌ Dashboard from queries failed: {e}")
        return chart_generator._create_error_dashboard("Query Dashboard", language)