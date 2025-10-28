@echo off
echo AI Model Sentinel - Automated Backup System
echo.
:: Create backup with timestamp
python -c "from local_backup import backup_system; backup_system.create_backup()"
echo.
echo Backup completed successfully
echo Backups stored in: backups\
echo.
pause
