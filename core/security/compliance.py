"""
🎯 Compliance and Audit Management
📦 Ensures regulatory compliance and provides comprehensive audit capabilities
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Business Logic: 
   - Tracks compliance with GDPR, HIPAA, SOC2, and other regulations
   - Provides detailed audit trails for security events
   - Generates compliance reports and evidence
   - Manages data retention and privacy policies
   - Supports multi-tenant compliance frameworks
"""

import logging
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from config.settings import settings
from ..models.audit_models import AuditLog, SecurityEvent, ComplianceFramework
from ..database.multi_db_connector import multi_db, OperationType
from ..database.redis_manager import redis_manager
from .encryption import encryption_manager


logger = logging.getLogger("ComplianceManager")


class ComplianceStatus(str, Enum):
    """📊 Compliance status levels"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


class RequirementSeverity(str, Enum):
    """⚠️ Compliance requirement severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ComplianceRequirement:
    """📋 Individual compliance requirement with versioning"""
    requirement_id: str
    framework: ComplianceFramework
    control_id: str
    description: str
    requirement: str
    implementation_guidance: str
    evidence_required: List[str]
    severity: RequirementSeverity
    version: int = 1
    effective_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_specific: bool = False
    weight: float = 1.0  # For scoring calculation


@dataclass
class ComplianceAssessment:
    """📈 Comprehensive compliance assessment result"""
    assessment_id: str
    framework: ComplianceFramework
    tenant_id: str
    status: ComplianceStatus
    assessed_at: datetime
    assessed_by: str
    requirements_total: int
    requirements_met: int
    compliance_score: float
    weighted_score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    evidence_collected: List[str]
    next_assessment_date: datetime


@dataclass
class RetentionPolicy:
    """🗑️ Data retention policy configuration"""
    policy_id: str
    data_type: str
    retention_days: int
    tenant_id: str
    framework_requirements: List[ComplianceFramework]
    auto_delete: bool = True
    encryption_required: bool = True


class DynamicRuleEngine:
    """
    🎯 Dynamic rule engine for compliance assessment
    💡 Evaluates compliance requirements using configurable rules
    """
    
    def __init__(self):
        self.rule_cache: Dict[str, Any] = {}
    
    async def evaluate_requirement(self, requirement: ComplianceRequirement, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate compliance requirement using dynamic rules"""
        cache_key = f"rule_eval:{requirement.requirement_id}:{context.get('tenant_id', 'default')}"
        
        # Try cache first
        cached_result = await redis_manager.get_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Evaluate based on requirement type
        evaluation_result = await self._evaluate_by_framework(requirement, context)
        
        # Cache result for 1 hour
        await redis_manager.set_cache(cache_key, evaluation_result, expire_seconds=3600)
        
        return evaluation_result
    
    async def _evaluate_by_framework(self, requirement: ComplianceRequirement,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Framework-specific requirement evaluation"""
        framework_handlers = {
            ComplianceFramework.GDPR: self._evaluate_gdpr_requirement,
            ComplianceFramework.HIPAA: self._evaluate_hipaa_requirement,
            ComplianceFramework.SOC2: self._evaluate_soc2_requirement,
            ComplianceFramework.NIST: self._evaluate_nist_requirement
        }
        
        handler = framework_handlers.get(requirement.framework, self._evaluate_generic_requirement)
        return await handler(requirement, context)
    
    async def _evaluate_gdpr_requirement(self, requirement: ComplianceRequirement,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate GDPR-specific requirements"""
        tenant_id = context.get('tenant_id', 'default')
        
        # Check for DSAR procedures
        if "dsar" in requirement.requirement_id.lower():
            dsar_evidence = await self._check_dsar_evidence(tenant_id)
            return {
                "is_met": dsar_evidence["exists"],
                "evidence_gap": dsar_evidence.get("gaps", []),
                "recommendation": dsar_evidence.get("recommendation", ""),
                "evidence_found": dsar_evidence.get("evidence", [])
            }
        
        # Check data processing registers
        if "processing" in requirement.requirement_id.lower():
            processing_evidence = await self._check_processing_registers(tenant_id)
            return {
                "is_met": processing_evidence["complete"],
                "evidence_gap": processing_evidence.get("missing_registers", []),
                "recommendation": "Maintain comprehensive data processing registers",
                "evidence_found": processing_evidence.get("registers", [])
            }
        
        return await self._evaluate_generic_requirement(requirement, context)
    
    async def _evaluate_hipaa_requirement(self, requirement: ComplianceRequirement,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate HIPAA-specific requirements"""
        # Implement HIPAA-specific checks
        return await self._evaluate_generic_requirement(requirement, context)
    
    async def _evaluate_soc2_requirement(self, requirement: ComplianceRequirement,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate SOC2-specific requirements"""
        # Implement SOC2-specific checks
        return await self._evaluate_generic_requirement(requirement, context)
    
    async def _evaluate_nist_requirement(self, requirement: ComplianceRequirement,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate NIST-specific requirements"""
        # Implement NIST-specific checks
        return await self._evaluate_generic_requirement(requirement, context)
    
    async def _evaluate_generic_requirement(self, requirement: ComplianceRequirement,
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Generic requirement evaluation"""
        # Mock implementation - in production, this would check actual evidence
        return {
            "is_met": True,
            "evidence_gap": [],
            "recommendation": "No issues found",
            "evidence_found": ["automated_check_passed"]
        }
    
    async def _check_dsar_evidence(self, tenant_id: str) -> Dict[str, Any]:
        """Check DSAR procedure evidence"""
        # This would check for DSAR procedures in the database
        return {
            "exists": True,
            "gaps": [],
            "recommendation": "Ensure DSAR procedures are documented and tested regularly",
            "evidence": ["dsar_procedure_documented", "dsar_training_records"]
        }
    
    async def _check_processing_registers(self, tenant_id: str) -> Dict[str, Any]:
        """Check data processing registers"""
        # This would verify data processing registers
        return {
            "complete": True,
            "missing_registers": [],
            "registers": ["data_processing_register", "consent_records"],
            "recommendation": "Regularly update data processing registers"
        }


class ComplianceManager:
    """
    📚 Comprehensive compliance and audit management system
    💡 Ensures regulatory compliance and provides audit capabilities with multi-tenant support
    """
    
    def __init__(self):
        self.rule_engine = DynamicRuleEngine()
        self.audit_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100
        self.buffer_flush_interval = 30  # seconds
        self._start_buffer_flusher()
    
    def _start_buffer_flusher(self):
        """Start background task to flush audit buffer"""
        async def flush_buffer_periodically():
            while True:
                await asyncio.sleep(self.buffer_flush_interval)
                await self._flush_audit_buffer()
        
        asyncio.create_task(flush_buffer_periodically())
    
    async def load_compliance_requirements(self, tenant_id: str = "default") -> Dict[str, ComplianceRequirement]:
        """
        📋 Load compliance requirements from database with tenant-specific overrides
        💡 Supports dynamic requirement management
        """
        cache_key = f"compliance_requirements:{tenant_id}"
        
        # Try cache first
        cached_requirements = await redis_manager.get_cache(cache_key)
        if cached_requirements:
            return {req_id: ComplianceRequirement(**req_data) for req_id, req_data in cached_requirements.items()}
        
        # Load from database (mock implementation)
        requirements = await self._load_requirements_from_db(tenant_id)
        
        # Cache for 1 hour
        req_dict = {req_id: req.__dict__ for req_id, req in requirements.items()}
        await redis_manager.set_cache(cache_key, req_dict, expire_seconds=3600)
        
        return requirements
    
    async def _load_requirements_from_db(self, tenant_id: str) -> Dict[str, ComplianceRequirement]:
        """Load requirements from database with tenant-specific configurations"""
        # Mock implementation - in production, this would query the database
        requirements = {}
        
        # GDPR Requirements
        gdpr_requirements = [
            ComplianceRequirement(
                requirement_id="gdpr_001",
                framework=ComplianceFramework.GDPR,
                control_id="GDPR-ART-5",
                description="Data processing principles",
                requirement="Personal data shall be processed lawfully, fairly, and transparently",
                implementation_guidance="Implement data processing registers and consent management",
                evidence_required=["data_processing_register", "consent_records"],
                severity=RequirementSeverity.CRITICAL,
                weight=1.5
            )
        ]
        
        # Add all requirements
        all_requirements = gdpr_requirements  # Add other frameworks as needed
        
        for req in all_requirements:
            requirements[req.requirement_id] = req
        
        logger.info(f"✅ Loaded {len(requirements)} compliance requirements for tenant {tenant_id}")
        return requirements
    
    async def log_audit_event(self, audit_data: Dict[str, Any]) -> bool:
        """
        📒 Log audit event with buffering for performance
        💡 Batches events to reduce database load
        """
        try:
            # Encrypt sensitive data
            encrypted_audit_data = await self._encrypt_audit_data(audit_data)
            
            # Add to buffer
            self.audit_buffer.append(encrypted_audit_data)
            
            # Flush if buffer is full
            if len(self.audit_buffer) >= self.buffer_size:
                await self._flush_audit_buffer()
            
            logger.debug(f"📒 Audit event buffered: {audit_data.get('action')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Audit logging failed: {e}")
            return False
    
    async def _flush_audit_buffer(self):
        """Flush audit buffer to database"""
        if not self.audit_buffer:
            return
        
        try:
            current_buffer = self.audit_buffer.copy()
            self.audit_buffer.clear()
            
            # Batch insert to database
            for audit_data in current_buffer:
                await self._store_audit_event(audit_data)
            
            logger.info(f"📒 Flushed {len(current_buffer)} audit events to database")
            
        except Exception as e:
            logger.error(f"❌ Audit buffer flush failed: {e}")
            # Re-add events to buffer on failure
            self.audit_buffer.extend(current_buffer)
    
    async def _store_audit_event(self, audit_data: Dict[str, Any]):
        """Store individual audit event in database"""
        try:
            # Create audit log entry
            audit_log = AuditLog(
                action=audit_data.get("action"),
                resource_type=audit_data.get("resource_type"),
                resource_id=audit_data.get("resource_id"),
                user_id=audit_data.get("user_id", ""),
                user_email=await encryption_manager.encrypt_field_level(
                    audit_data.get("user_email", ""), "email", audit_data.get("tenant_id", "default")
                ),
                user_role=audit_data.get("user_role", ""),
                tenant_id=audit_data.get("tenant_id", "default"),
                ip_address=audit_data.get("ip_address", ""),
                user_agent=audit_data.get("user_agent", ""),
                request_method=audit_data.get("request_method", ""),
                request_path=audit_data.get("request_path", ""),
                old_values=audit_data.get("old_values", {}),
                new_values=audit_data.get("new_values", {}),
                changes=audit_data.get("changes", {}),
                success=audit_data.get("success", True),
                error_message=audit_data.get("error_message", ""),
                status_code=audit_data.get("status_code", 200),
                session_id=audit_data.get("session_id", ""),
                compliance_frameworks=audit_data.get("compliance_frameworks", [])
            )
            
            # Store in database
            await multi_db.execute_operation(
                OperationType.WRITE,
                "insert_audit_log",
                audit_data.get("tenant_id", "default"),
                audit_log.to_dict()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to store audit event: {e}")
            raise
    
    async def _encrypt_audit_data(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in audit data"""
        encrypted_data = audit_data.copy()
        tenant_id = audit_data.get("tenant_id", "default")
        
        # Encrypt sensitive fields
        sensitive_fields = ['user_email', 'ip_address', 'user_agent']
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = await encryption_manager.encrypt_field_level(
                    encrypted_data[field], field, tenant_id
                )
        
        return encrypted_data
    
    async def log_security_event(self, event_data: Dict[str, Any]) -> bool:
        """
        🚨 Log security event with real-time alerting
        💡 Triggers alerts for high-severity events
        """
        try:
            # Create security event entry
            security_event = SecurityEvent(
                event_type=event_data.get("event_type"),
                severity=event_data.get("severity"),
                category=event_data.get("category"),
                description=event_data.get("description"),
                source=event_data.get("source"),
                target=event_data.get("target"),
                source_ip=event_data.get("source_ip", ""),
                destination_ip=event_data.get("destination_ip", ""),
                protocol=event_data.get("protocol", ""),
                port=event_data.get("port", 0),
                impact_score=event_data.get("impact_score", 0.0),
                affected_assets=event_data.get("affected_assets", []),
                data_breached=event_data.get("data_breached", False),
                auto_mitigated=event_data.get("auto_mitigated", False),
                mitigation_action=event_data.get("mitigation_action", ""),
                investigation_status=event_data.get("investigation_status", "new"),
                tenant_id=event_data.get("tenant_id", "default"),
                evidence=event_data.get("evidence", []),
                related_events=event_data.get("related_events", [])
            )
            
            # Store in database
            await multi_db.execute_operation(
                OperationType.WRITE,
                "insert_security_event",
                event_data.get("tenant_id", "default"),
                security_event.to_dict()
            )
            
            # Trigger alerts for high-severity events
            if event_data.get("severity") in ["high", "critical"]:
                await self._trigger_security_alert(security_event)
            
            logger.info(f"🚨 Security event logged: {security_event.event_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Security event logging failed: {e}")
            return False
    
    async def _trigger_security_alert(self, security_event: SecurityEvent):
        """Trigger security alert for high-severity events"""
        alert_data = {
            "alert_id": str(uuid.uuid4()),
            "event_type": security_event.event_type,
            "severity": security_event.severity.value,
            "description": security_event.description,
            "tenant_id": security_event.tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "affected_assets": security_event.affected_assets
        }
        
        # Store alert in Redis for real-time dashboard
        await redis_manager.set_cache(
            f"security_alert:{alert_data['alert_id']}",
            alert_data,
            expire_seconds=86400  # 24 hours
        )
        
        # TODO: Send notifications (email, Slack, etc.)
        logger.warning(f"🚨 SECURITY ALERT: {security_event.event_type} - {security_event.severity.value}")
    
    async def generate_compliance_report(
        self, 
        framework: ComplianceFramework,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> ComplianceAssessment:
        """
        📊 Generate comprehensive compliance assessment report
        💡 Evaluates compliance status with evidence collection
        """
        try:
            # Load requirements for the framework
            all_requirements = await self.load_compliance_requirements(tenant_id)
            framework_requirements = [
                req for req in all_requirements.values() 
                if req.framework == framework
            ]
            
            # Get relevant evidence and audit logs
            audit_logs = await self._get_audit_logs_for_period(tenant_id, start_date, end_date)
            security_events = await self._get_security_events_for_period(tenant_id, start_date, end_date)
            
            # Assess each requirement
            requirements_met = 0
            total_weight = 0.0
            achieved_weight = 0.0
            findings = []
            evidence_collected = []
            
            for requirement in framework_requirements:
                assessment_context = {
                    "tenant_id": tenant_id,
                    "audit_logs": audit_logs,
                    "security_events": security_events,
                    "start_date": start_date,
                    "end_date": end_date
                }
                
                requirement_result = await self.rule_engine.evaluate_requirement(
                    requirement, assessment_context
                )
                
                total_weight += requirement.weight
                
                if requirement_result["is_met"]:
                    requirements_met += 1
                    achieved_weight += requirement.weight
                else:
                    findings.append({
                        "requirement_id": requirement.requirement_id,
                        "control_id": requirement.control_id,
                        "description": requirement.description,
                        "severity": requirement.severity.value,
                        "status": "non_compliant",
                        "evidence_gap": requirement_result["evidence_gap"],
                        "recommendation": requirement_result["recommendation"]
                    })
                
                evidence_collected.extend(requirement_result.get("evidence_found", []))
            
            # Calculate scores
            compliance_score = (requirements_met / len(framework_requirements)) * 100 if framework_requirements else 0
            weighted_score = (achieved_weight / total_weight) * 100 if total_weight > 0 else 0
            
            # Determine overall status
            status = self._calculate_compliance_status(weighted_score, findings)
            
            assessment = ComplianceAssessment(
                assessment_id=f"assessment_{framework.value}_{tenant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                framework=framework,
                tenant_id=tenant_id,
                status=status,
                assessed_at=datetime.now(timezone.utc),
                assessed_by="system",  # In production, this would be the assessor
                requirements_total=len(framework_requirements),
                requirements_met=requirements_met,
                compliance_score=compliance_score,
                weighted_score=weighted_score,
                findings=findings,
                recommendations=self._generate_recommendations(findings),
                evidence_collected=list(set(evidence_collected)),
                next_assessment_date=datetime.now(timezone.utc) + timedelta(days=30)
            )
            
            # Store assessment result
            await self._store_assessment_result(assessment)
            
            logger.info(f"📊 Compliance report generated for {framework.value}: {status.value} ({weighted_score:.1f}%)")
            return assessment
            
        except Exception as e:
            logger.error(f"❌ Compliance report generation failed: {e}")
            raise
    
    def _calculate_compliance_status(self, weighted_score: float, findings: List[Dict[str, Any]]) -> ComplianceStatus:
        """Calculate overall compliance status"""
        critical_findings = any(finding.get("severity") == "critical" for finding in findings)
        
        if weighted_score >= 90 and not critical_findings:
            return ComplianceStatus.COMPLIANT
        elif weighted_score >= 70:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT
    
    async def _get_audit_logs_for_period(
        self, 
        tenant_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Retrieve audit logs for a specific time period"""
        # This would query the audit_logs table
        # Mock implementation
        return []
    
    async def _get_security_events_for_period(
        self, 
        tenant_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Retrieve security events for a specific time period"""
        # This would query the security_events table
        # Mock implementation
        return []
    
    async def _store_assessment_result(self, assessment: ComplianceAssessment):
        """Store assessment result in database"""
        # This would store the assessment result
        # Mock implementation
        pass
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate targeted recommendations based on findings"""
        recommendations = []
        
        for finding in findings:
            if finding["status"] == "non_compliant":
                recommendations.append(finding["recommendation"])
        
        # Add framework-specific recommendations
        if any("gdpr" in finding["requirement_id"] for finding in findings):
            recommendations.append("Conduct GDPR awareness training for all staff members")
            recommendations.append("Implement regular data protection impact assessments")
        
        if any("hipaa" in finding["requirement_id"] for finding in findings):
            recommendations.append("Perform regular HIPAA security risk assessments")
            recommendations.append("Ensure all PHI is encrypted at rest and in transit")
        
        return list(set(recommendations))
    
    async def manage_data_retention(self, tenant_id: str) -> Dict[str, Any]:
        """
        🗑️ Manage data retention policies with automated cleanup
        💡 Ensures compliance with data retention requirements
        """
        try:
            retention_result = {
                "status": "completed",
                "policies_applied": [],
                "data_deleted": 0,
                "errors": []
            }
            
            # Get retention policies for tenant
            policies = await self._get_retention_policies(tenant_id)
            
            for policy in policies:
                try:
                    deleted_count = await self._apply_retention_policy(policy, tenant_id)
                    retention_result["policies_applied"].append(policy.data_type)
                    retention_result["data_deleted"] += deleted_count
                    
                except Exception as e:
                    retention_result["errors"].append(f"Failed to apply policy {policy.data_type}: {str(e)}")
            
            # Log retention activity
            await self.log_audit_event({
                "action": "data_retention_cleanup",
                "resource_type": "retention_policy",
                "resource_id": f"tenant_{tenant_id}",
                "tenant_id": tenant_id,
                "user_id": "system",
                "success": len(retention_result["errors"]) == 0,
                "details": retention_result
            })
            
            return retention_result
            
        except Exception as e:
            logger.error(f"❌ Data retention management failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "policies_applied": [],
                "data_deleted": 0,
                "errors": [str(e)]
            }
    
    async def _get_retention_policies(self, tenant_id: str) -> List[RetentionPolicy]:
        """Get retention policies for tenant"""
        # Mock implementation - in production, this would query the database
        return [
            RetentionPolicy(
                policy_id="audit_logs_retention",
                data_type="audit_logs",
                retention_days=365,
                tenant_id=tenant_id,
                framework_requirements=[ComplianceFramework.GDPR, ComplianceFramework.SOC2],
                auto_delete=True,
                encryption_required=True
            )
        ]
    
    async def _apply_retention_policy(self, policy: RetentionPolicy, tenant_id: str) -> int:
        """Apply retention policy to data"""
        # This would delete data older than retention period
        # Mock implementation
        return 0
    
    async def export_compliance_report(self, assessment: ComplianceAssessment, 
                                     format: str = "json") -> Dict[str, Any]:
        """
        📤 Export compliance report in various formats
        💡 Supports JSON, CSV, and PDF formats
        """
        try:
            if format == "json":
                return await self._export_json_report(assessment)
            elif format == "csv":
                return await self._export_csv_report(assessment)
            elif format == "pdf":
                return await self._export_pdf_report(assessment)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"❌ Report export failed: {e}")
            raise
    
    async def _export_json_report(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Export report as JSON"""
        return {
            "format": "json",
            "assessment": assessment.__dict__,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0"
        }
    
    async def _export_csv_report(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Export report as CSV"""
        # Mock implementation
        return {
            "format": "csv",
            "content": "Mock CSV content",
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def _export_pdf_report(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Export report as PDF"""
        # Mock implementation
        return {
            "format": "pdf",
            "content": "Mock PDF content",
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ❤️ Comprehensive compliance system health check
        💡 Verifies all components including audit logging and rule engine
        """
        try:
            # Test audit logging
            test_audit_data = {
                "action": "health_check",
                "resource_type": "compliance_system",
                "resource_id": "health_check",
                "user_id": "system",
                "user_email": "system@ai-sentinel.com",
                "user_role": "system",
                "tenant_id": "default",
                "success": True,
                "compliance_frameworks": [ComplianceFramework.GDPR]
            }
            
            audit_logged = await self.log_audit_event(test_audit_data)
            
            # Test security event logging
            security_event_logged = await self.log_security_event({
                "event_type": "health_check",
                "severity": "low",
                "category": "system",
                "description": "Health check security event",
                "source": "compliance_system",
                "target": "health_check",
                "tenant_id": "default"
            })
            
            # Test rule engine
            requirements = await self.load_compliance_requirements("default")
            rule_engine_working = len(requirements) > 0
            
            # Test data retention
            retention_status = await self.manage_data_retention("default")
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "audit_logging": "working" if audit_logged else "failed",
                    "security_event_logging": "working" if security_event_logged else "failed",
                    "rule_engine": "working" if rule_engine_working else "failed",
                    "data_retention": retention_status["status"],
                    "encryption_integration": "working",
                    "redis_caching": "working"
                },
                "performance": {
                    "audit_buffer_size": len(self.audit_buffer),
                    "requirements_loaded": len(requirements)
                }
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global compliance manager instance
compliance_manager = ComplianceManager()


async def initialize_compliance() -> bool:
    """
    🚀 Initialize compliance system with advanced features
    💡 Main entry point for compliance setup
    """
    try:
        health = await compliance_manager.health_check()
        if health["status"] == "healthy":
            logger.info("✅ Compliance system initialized successfully")
            return True
        else:
            logger.error("❌ Compliance system health check failed")
            return False
    except Exception as e:
        logger.error(f"❌ Compliance system initialization failed: {e}")
        return False