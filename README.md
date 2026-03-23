# ParkGuard

## Quick Start

To seed the database with initial test data, run the following command in your terminal (using PowerShell/CMD):

```powershell
Get-Content seed/parking_data.sql | docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard
```
*(For bash/Linux/macOS, you can use: `docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard < seed/parking_data.sql`)*

## Reset DB

If you need to wipe the database and start fresh with a clean schema and seed data, follow these steps:

1. **Drop the Schema**
   This will completely drop all existing tables and data:
   ```bash
   docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
   ```

2. **Recreate Tables (`create_all`)**
   Restart your backend server (or let it auto-reload if you're using `uvicorn --reload`). The application's `lifespan` startup routine will execute SQLAlchemy's `create_all` to recreate the empty tables according to your models.

3. **Re-Seed Data**
   Run the seed command again to insert the initial data:
   ```powershell
   Get-Content seed/parking_data.sql | docker exec -i -e PGPASSWORD=parkguard parkguard-db psql -U parkguard -d parkguard
   ```