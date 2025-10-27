# User Models - Complete Version
class UserRole:
    SUPER_ADMIN = "super_admin"
    DEVELOPER = "developer"
    ADMIN = "admin"
    USER = "user"
    SECURITY_ANALYST = "security_analyst"
    TENANT_ADMIN = "tenant_admin"
    AUDITOR = "auditor"

class UserStatus:
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class User:
    def __init__(self, username, role):
        self.username = username
        self.role = role
    
    def to_dict(self):
        return {"username": self.username, "role": self.role}

print('User models complete')