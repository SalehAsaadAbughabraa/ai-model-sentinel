#!/usr/bin/env python3
"""
🚀 AI Model Sentinel v2.0.0 - Enhanced Startup Script
📦 Advanced script to start FastAPI with multi-environment support
👨‍💻 Author: Saleh Abughabraa
💡 Features:
   - Multi-environment configuration (dev, staging, production)
   - Pre-flight health checks and dependency validation
   - Colorized logging and comprehensive status reporting
   - CI/CD integration with JSON output support
"""

import uvicorn
import os
import sys
import asyncio
import importlib
import pkg_resources
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import json
from datetime import datetime

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def colorize(text: str, color: str) -> str:
    """Add color to terminal text"""
    return f"{color}{text}{Colors.END}"

class SentinelStartupManager:
    """Enhanced startup manager for AI Model Sentinel"""
    
    def __init__(self, environment: str = None, headless: bool = False):
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.headless = headless
        self.start_time = datetime.now()
        self.health_status = {}
        
    def print_banner(self):
        """Print application banner"""
        if self.headless:
            return
            
        banner = f"""
{Colors.CYAN}{Colors.BOLD}
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🛡️  AI Model Sentinel v2.0.0 - Security Analytics Platform  ║
    ║                                                              ║
    ║   👨‍💻 Developer: Saleh Abughabraa                            ║
    ║   🌍 Environment: {self.environment.upper():<20}                 ║
    ║   🕐 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
{Colors.END}
        """
        print(banner)
    
    async def validate_dependencies(self) -> bool:
        """Validate all required dependencies"""
        if not self.headless:
            print(colorize("🔍 Validating dependencies...", Colors.BLUE))
        
        critical_deps = {
            "fastapi": "0.104.1",
            "uvicorn": "0.24.0",
            "pydantic": "2.5.0",
            "sqlalchemy": "2.0.23",
            "asyncpg": "0.29.0",
            "redis": "5.0.1",
        }
        
        missing_deps = []
        version_warnings = []
        
        for dep, min_version in critical_deps.items():
            try:
                spec = importlib.util.find_spec(dep)
                if spec is None:
                    missing_deps.append(dep)
                else:
                    installed_version = pkg_resources.get_distribution(dep).version
                    if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                        version_warnings.append(f"{dep} (installed: {installed_version}, required: {min_version})")
            except Exception:
                missing_deps.append(dep)
        
        if missing_deps:
            if not self.headless:
                print(colorize(f"❌ Missing dependencies: {', '.join(missing_deps)}", Colors.RED))
                print(colorize("💡 Run: pip install -r requirements.txt", Colors.YELLOW))
            return False
        
        if version_warnings and not self.headless:
            for warning in version_warnings:
                print(colorize(f"⚠️  Version warning: {warning}", Colors.YELLOW))
        
        if not self.headless:
            print(colorize("✅ All dependencies validated", Colors.GREEN))
        
        return True
    
    async def check_database_connections(self) -> bool:
        """Check database and cache connections"""
        if not self.headless:
            print(colorize("🗄️  Checking database connections...", Colors.BLUE))
        
        try:
            # Check PostgreSQL
            from core.database.multi_db_connector import multi_db
            async with multi_db.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    if not self.headless:
                        print(colorize("❌ PostgreSQL test query failed", Colors.RED))
                    return False
            
            # Check Redis
            import redis
            from config.settings import get_settings
            settings = get_settings()
            
            redis_client = redis.from_url(settings.REDIS_URL)
            redis_client.ping()
            
            if not self.headless:
                print(colorize("✅ Database connections established", Colors.GREEN))
            return True
            
        except Exception as e:
            if not self.headless:
                print(colorize(f"❌ Database connection failed: {e}", Colors.RED))
            return False
    
    async def validate_application_structure(self) -> bool:
        """Validate application module structure"""
        if not self.headless:
            print(colorize("📦 Validating application structure...", Colors.BLUE))
        
        try:
            from web_interface.app import app
            
            # Check if essential routers are mounted
            expected_routes = [
                "/api/v1/scans",
                "/api/v1/auth", 
                "/api/v1/audit",
                "/api/v1/health",
                "/api/v1/users",
                "/api/v1/analytics"
            ]
            
            mounted_routes = []
            for route in app.routes:
                if hasattr(route, 'path'):
                    mounted_routes.append(route.path)
            
            missing_routes = []
            for expected in expected_routes:
                if not any(route.startswith(expected) for route in mounted_routes):
                    missing_routes.append(expected)
            
            if missing_routes:
                if not self.headless:
                    print(colorize(f"⚠️  Missing routes: {missing_routes}", Colors.YELLOW))
                # Don't fail startup for missing routes, just warn
            else:
                if not self.headless:
                    print(colorize("✅ All essential routes are mounted", Colors.GREEN))
            
            return True
            
        except Exception as e:
            if not self.headless:
                print(colorize(f"❌ Application structure validation failed: {e}", Colors.RED))
            return False
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks"""
        health_checks = {
            "dependencies": await self.validate_dependencies(),
            "database": await self.check_database_connections(),
            "application_structure": await self.validate_application_structure(),
        }
        
        all_healthy = all(health_checks.values())
        health_checks["overall"] = all_healthy
        
        self.health_status = health_checks
        return health_checks
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration based on environment"""
        config = {
            "host": os.getenv("API_HOST", "0.0.0.0"),
            "port": int(os.getenv("API_PORT", "8000")),
            "workers": int(os.getenv("API_WORKERS", "1")),
            "reload": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "info"),
        }
        
        # Environment-specific adjustments
        if self.environment == "production":
            config["reload"] = False  # Never reload in production
            if config["workers"] < 2:
                config["workers"] = 2  # Minimum 2 workers in production
            config["log_level"] = "warning"
            
        elif self.environment == "staging":
            config["reload"] = False
            config["log_level"] = "info"
            
        else:  # development
            config["reload"] = True
            config["workers"] = 1  # Single worker with reload
            
        return config
    
    def print_server_info(self, config: Dict[str, Any]):
        """Print server configuration information"""
        if self.headless:
            return
            
        print(colorize("\n🌐 Server Configuration:", Colors.CYAN))
        print(f"   • Host: {config['host']}")
        print(f"   • Port: {config['port']}")
        print(f"   • Workers: {config['workers']}")
        print(f"   • Reload: {config['reload']}")
        print(f"   • Log Level: {config['log_level']}")
        
        print(colorize("\n📚 API Documentation:", Colors.CYAN))
        host_display = "localhost" if config["host"] == "0.0.0.0" else config["host"]
        print(f"   • Swagger UI: http://{host_display}:{config['port']}/docs")
        print(f"   • ReDoc: http://{host_display}:{config['port']}/redoc")
        print(f"   • Health Check: http://{host_display}:{config['port']}/api/v1/health")
        
        print(colorize("\n🚀 Starting server...", Colors.GREEN))
    
    async def start_application(self, run_health_check: bool = True):
        """Start the FastAPI application with comprehensive setup"""
        
        # Print banner
        self.print_banner()
        
        # Run health checks if requested
        if run_health_check:
            health_status = await self.run_health_checks()
            
            if not health_status["overall"]:
                if not self.headless:
                    print(colorize("\n❌ Health checks failed. Application cannot start.", Colors.RED))
                sys.exit(1)
        
        # Get server configuration
        config = self.get_server_config()
        
        # Print server information
        self.print_server_info(config)
        
        # Import application
        try:
            from web_interface.app import app
        except ImportError as e:
            if not self.headless:
                print(colorize(f"❌ Failed to import application: {e}", Colors.RED))
            sys.exit(1)
        
        # Start Uvicorn server
        uvicorn.run(
            "main:app",
            host=config["host"],
            port=config["port"],
            workers=config["workers"] if not config["reload"] else 1,
            reload=config["reload"],
            log_level=config["log_level"],
            access_log=not self.headless
        )

async def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='AI Model Sentinel Startup Manager')
    parser.add_argument('--environment', '-e', 
                       choices=['development', 'staging', 'production'],
                       help='Environment to run in')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no console output)')
    parser.add_argument('--skip-health-check', action='store_true',
                       help='Skip pre-flight health checks')
    parser.add_argument('--health-check-only', action='store_true',
                       help='Only run health checks and exit')
    parser.add_argument('--output-format', choices=['human', 'json'],
                       default='human', help='Output format for health checks')
    
    args = parser.parse_args()
    
    # Initialize startup manager
    startup_manager = SentinelStartupManager(
        environment=args.environment,
        headless=args.headless
    )
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.append(str(current_dir))
    
    # Health check only mode
    if args.health_check_only:
        health_status = await startup_manager.run_health_checks()
        
        if args.output_format == 'json':
            print(json.dumps({
                "system": "AI Model Sentinel v2.0.0",
                "environment": startup_manager.environment,
                "timestamp": datetime.now().isoformat(),
                "health_checks": health_status,
                "status": "healthy" if health_status["overall"] else "unhealthy"
            }, indent=2))
        else:
            if health_status["overall"]:
                print(colorize("✅ All health checks passed", Colors.GREEN))
            else:
                print(colorize("❌ Health checks failed", Colors.RED))
                sys.exit(1)
        return
    
    # Start the application
    await startup_manager.start_application(run_health_check=not args.skip_health_check)

if __name__ == "__main__":
    asyncio.run(main())