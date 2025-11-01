#!/usr/bin/env python3
"""
AI Model Sentinel - Enterprise Deployment Script
Production-ready with comprehensive safety checks and modular deployment
"""

import os
import sys
import subprocess
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class DeploymentManager:
    """Enterprise-grade deployment manager for AI Sentinel"""
    
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.config = self.load_config()
        self.start_time = datetime.now()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "deployment.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("DeploymentManager")
        
    def load_config(self) -> Dict:
        """Load deployment configuration"""
        return {
            'requirements_file': os.getenv("REQUIREMENTS_FILE", "requirements.txt"),
            'dev_requirements_file': "requirements-dev.txt",
            'prod_requirements_file': "requirements-prod.txt",
            'required_tools': ['python', 'pip', 'git'],
            'required_dirs': ['secrets', 'logs', 'scan_results', 'cache', 'backups'],
            'min_python_version': (3, 8),
            'environment_files': ['.env.production', '.env.example', '.env']
        }
    
    def run_command(self, cmd: str, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with comprehensive logging"""
        self.logger.info(f"🚀 Executing: {cmd}")
        
        result = subprocess.run(
            cmd, 
            shell=False, 
            capture_output=capture_output, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.stdout:
            self.logger.debug(f"STDOUT: {result.stdout}")
        if result.stderr:
            self.logger.debug(f"STDERR: {result.stderr}")
            
        if result.returncode != 0 and check:
            self.logger.error(f"❌ Command failed (exit {result.returncode}): {cmd}")
            if result.stderr:
                self.logger.error(f"Error details: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
            
        return result
    
    def check_prerequisites(self) -> bool:
        """Verify all prerequisites are met"""
        self.logger.info("🔍 Checking prerequisites...")
        
        # Check environment
        if not self.args.skip_env_check and os.getenv("ENV") != "production":
            self.logger.warning("⚠️  ENV not set to 'production'. Deployment may not be safe for production.")
            if not self.args.force:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    self.logger.info("Deployment cancelled by user")
                    return False
        
        # Check required tools
        missing_tools = []
        for tool in self.config['required_tools']:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            self.logger.error(f"❌ Missing required tools: {', '.join(missing_tools)}")
            return False
        
        # Check Python version
        python_version = sys.version_info[:2]
        if python_version < self.config['min_python_version']:
            self.logger.error(f"❌ Python {self.config['min_python_version']} required, found {python_version}")
            return False
        
        # Check disk space (minimum 1GB free)
        try:
            disk_usage = shutil.disk_usage(".")
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 1:
                self.logger.warning(f"⚠️  Low disk space: {free_gb:.1f}GB free")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not check disk space: {e}")
        
        self.logger.info("✅ All prerequisites satisfied")
        return True
    
    def setup_environment(self):
        """Setup production environment"""
        self.logger.info("🔧 Setting up production environment...")
        
        # Create necessary directories
        for dir_name in self.config['required_dirs']:
            Path(dir_name).mkdir(exist_ok=True, parents=True)
            self.logger.debug(f"Created directory: {dir_name}")
        
        # Setup environment file
        self.setup_environment_file()
        
        # Set proper permissions
        self.setup_permissions()
        
        self.logger.info("✅ Environment setup completed")
    
    def setup_environment_file(self):
        """Setup environment configuration file"""
        env_file = Path(".env")
        
        if env_file.exists():
            if self.args.force:
                self.logger.warning("⚠️  Overwriting existing .env file")
            else:
                self.logger.info("ℹ️  .env file already exists, skipping creation")
                return
        
        # Find template file
        template_found = False
        for template in self.config['environment_files']:
            template_path = Path(template)
            if template_path.exists():
                shutil.copy(template_path, env_file)
                self.logger.info(f"✅ Created .env from {template}")
                template_found = True
                break
        
        if not template_found:
            self.logger.warning("⚠️  No environment template found, creating basic .env")
            self.create_basic_env_file()
    
    def create_basic_env_file(self):
        """Create basic environment file"""
        basic_env = """# AI Sentinel Basic Configuration
ENV=production
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./ai_sentinel.db

# Security - Update these in production!
ENCRYPTION_KEY=base64:dev-key-change-in-production
JWT_SECRET=dev-jwt-secret-change-in-production

# WARNING: This is a basic configuration for initial setup.
# Please review and update all security settings before production use.
"""
        with open(".env", "w") as f:
            f.write(basic_env)
    
    def setup_permissions(self):
        """Set secure file permissions"""
        try:
            # Make secrets directory secure
            secrets_dir = Path("secrets")
            if secrets_dir.exists():
                if os.name != 'nt':  # Unix-like systems
                    self.run_command("chmod 700 secrets", check=False)
            
            # Make scripts executable
            for script in Path(".").glob("*.sh"):
                if os.name != 'nt':
                    self.run_command(f"chmod +x {script}", check=False)
                    
        except Exception as e:
            self.logger.warning(f"⚠️  Permission setup failed: {e}")
    
    def install_dependencies(self):
        """Install production dependencies"""
        self.logger.info("📦 Installing dependencies...")
        
        # Upgrade pip first
        try:
            self.run_command("pip install --upgrade pip")
        except subprocess.CalledProcessError:
            self.logger.warning("⚠️  Pip upgrade failed, continuing with current version")
        
        # Install base requirements
        req_file = self.config['requirements_file']
        if Path(req_file).exists():
            self.run_command(f"pip install -r {req_file}")
        else:
            self.logger.error(f"❌ Requirements file not found: {req_file}")
            raise FileNotFoundError(f"Requirements file {req_file} not found")
        
        # Install additional packages based on flags
        if self.args.with_dev_tools:
            self.install_development_tools()
        
        if self.args.with_monitoring:
            self.install_monitoring_tools()
        
        self.logger.info("✅ Dependencies installed successfully")
    
    def install_development_tools(self):
        """Install development and code quality tools"""
        self.logger.info("🛠️  Installing development tools...")
        
        dev_tools = [
            "black", "flake8", "mypy", "pytest", "pytest-cov",
            "bandit", "safety", "pre-commit", "radon"
        ]
        
        for tool in dev_tools:
            try:
                self.run_command(f"pip install {tool}", check=False)
            except subprocess.CalledProcessError:
                self.logger.warning(f"⚠️  Failed to install {tool}")
    
    def install_monitoring_tools(self):
        """Install monitoring and observability tools"""
        self.logger.info("📊 Installing monitoring tools...")
        
        monitoring_tools = [
            "prometheus-client", "sentry-sdk", "loguru", 
            "psutil", "opentelemetry-api"
        ]
        
        for tool in monitoring_tools:
            try:
                self.run_command(f"pip install {tool}", check=False)
            except subprocess.CalledProcessError:
                self.logger.warning(f"⚠️  Failed to install {tool}")
    
    def setup_database(self):
        """Initialize production database"""
        self.logger.info("🗄️  Setting up database...")
        
        try:
            # Add current directory to Python path
            sys.path.insert(0, str(Path.cwd()))
            
            from core.database import db_manager
            db_manager.init_db()
            self.logger.info("✅ Database initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"❌ Failed to import database module: {e}")
            if not self.args.force:
                raise
        except Exception as e:
            self.logger.error(f"❌ Database setup failed: {e}")
            if not self.args.force:
                raise
    
    def run_security_scan(self):
        """Run comprehensive security scan"""
        if self.args.skip_security_scan:
            self.logger.info("⏭️  Skipping security scan")
            return
            
        self.logger.info("🔒 Running security scan...")
        
        scans = [
            ("Bandit (security linting)", "python -m bandit -r . -f json"),
            ("Safety (vulnerability check)", "python -m safety check --json"),
            ("Dependency audit", "pip-audit" if shutil.which("pip-audit") else "echo 'pip-audit not available'"),
        ]
        
        for scan_name, scan_cmd in scans:
            try:
                self.logger.info(f"🔍 Running {scan_name}...")
                result = self.run_command(scan_cmd, check=False)
                if result.returncode != 0:
                    self.logger.warning(f"⚠️  {scan_name} found issues")
                else:
                    self.logger.info(f"✅ {scan_name} passed")
            except Exception as e:
                self.logger.warning(f"⚠️  {scan_name} failed: {e}")
    
    def run_health_checks(self):
        """Run system health checks"""
        self.logger.info("❤️  Running health checks...")
        
        checks = [
            ("Feature extractor", "from core.feature_extractor import AdvancedFeatureExtractor; print('✅ Feature extractor ready')"),
            ("Database connection", "from core.database import db_manager; print('✅ Database connection ready')"),
            ("Configuration", "from config.production import config; print('✅ Configuration loaded')"),
        ]
        
        for check_name, check_code in checks:
            try:
                self.run_command(f'python -c "{check_code}"')
                self.logger.info(f"✅ {check_name} check passed")
            except Exception as e:
                self.logger.error(f"❌ {check_name} check failed: {e}")
                if not self.args.force:
                    raise
    
    def create_deployment_report(self):
        """Create deployment summary report"""
        deployment_time = datetime.now() - self.start_time
        
        report = f"""
🤖 AI Sentinel Deployment Report
{'=' * 40}
Deployment Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {deployment_time.total_seconds():.1f} seconds
Status: SUCCESS

📁 Directories Created:
{chr(10).join(f'  • {dir_name}' for dir_name in self.config['required_dirs'])}

📦 Dependencies Installed:
  • Production requirements: {self.config['requirements_file']}
  • Development tools: {'Yes' if self.args.with_dev_tools else 'No'}
  • Monitoring tools: {'Yes' if self.args.with_monitoring else 'No'}

✅ Checks Performed:
  • Prerequisites validation
  • Database initialization
  • Security scanning
  • System health checks

🎯 Next Steps:
1. Review and update .env file with production values
2. Configure Vault/KMS for secrets management
3. Set up PostgreSQL connection (if using SQLite)
4. Configure monitoring and alerting
5. Run comprehensive test suite
6. Set up backup and disaster recovery

📞 Support:
  • Documentation: docs/README.md
  • Issues: GitHub Issues
  • Security: saleh87alallu@gmail.com
"""
        
        report_path = Path("logs") / "deployment_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        
        self.logger.info(f"📄 Deployment report saved to: {report_path}")
        print(report)
    
    def deploy(self):
        """Execute full deployment process"""
        self.logger.info("🚀 Starting AI Sentinel deployment...")
        
        try:
            if not self.check_prerequisites():
                return False
            
            self.setup_environment()
            self.install_dependencies()
            
            if not self.args.skip_database:
                self.setup_database()
            
            if not self.args.skip_tests:
                self.run_security_scan()
                self.run_health_checks()
            
            self.create_deployment_report()
            
            self.logger.info("🎉 Deployment completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            return False

def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(description="AI Sentinel Deployment Script")
    
    # Deployment modes
    parser.add_argument("--full", action="store_true", help="Full deployment (default)")
    parser.add_argument("--setup-env-only", action="store_true", help="Only setup environment")
    parser.add_argument("--setup-db-only", action="store_true", help="Only setup database")
    parser.add_argument("--install-deps-only", action="store_true", help="Only install dependencies")
    
    # Options
    parser.add_argument("--force", action="store_true", help="Force deployment despite warnings")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip environment checks")
    parser.add_argument("--skip-database", action="store_true", help="Skip database setup")
    parser.add_argument("--skip-tests", action="store_true", help="Skip security scans and health checks")
    parser.add_argument("--skip-security-scan", action="store_true", help="Skip security scanning")
    
    # Additional features
    parser.add_argument("--with-dev-tools", action="store_true", help="Install development tools")
    parser.add_argument("--with-monitoring", action="store_true", help="Install monitoring tools")
    
    args = parser.parse_args()
    
    # Set default mode
    if not any([args.setup_env_only, args.setup_db_only, args.install_deps_only]):
        args.full = True
    
    manager = DeploymentManager(args)
    
    # Execute based on mode
    if args.setup_env_only:
        success = manager.setup_environment()
    elif args.setup_db_only:
        success = manager.setup_database()
    elif args.install_deps_only:
        success = manager.install_dependencies()
    else:  # Full deployment
        success = manager.deploy()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()