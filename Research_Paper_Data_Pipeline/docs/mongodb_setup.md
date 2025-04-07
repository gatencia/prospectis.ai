# MongoDB Setup for Prospectis

This document provides instructions for setting up MongoDB for the Prospectis research pipeline.

## Local Setup

### Installation

#### Ubuntu/Debian
```bash
# Import the public key
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -

# Create a list file for MongoDB
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Reload local package database
sudo apt-get update

# Install MongoDB
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod

# Enable MongoDB to start on boot
sudo systemctl enable mongod
```

#### macOS (with Homebrew)
```bash
# Install MongoDB
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB service
brew services start mongodb-community
```

#### Windows
1. Download the MongoDB installer from the [MongoDB Download Center](https://www.mongodb.com/try/download/community)
2. Run the installer and follow the installation wizard
3. MongoDB will be installed as a Windows service

### Verification

To verify that MongoDB is running correctly:

```bash
mongo --eval "db.version()"
```

You should see the MongoDB version printed to the console.

## Cloud Setup (MongoDB Atlas)

For production environments, we recommend using MongoDB Atlas, a fully managed cloud database service.

1. Create an account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a new cluster (the free tier is sufficient for testing)
3. Configure network access to allow connections from your application
4. Create a database user with appropriate permissions
5. Obtain the connection string and update your `.env` file

Example connection string format:
```
mongodb+srv://<username>:<password>@<cluster-url>/<database>?retryWrites=true&w=majority
```

## Database Configuration

### Setting Up Indexes

The research pipeline automatically creates necessary indexes when it runs. However, you can manually set up indexes using MongoDB Compass or the MongoDB shell:

```javascript
// Connect to MongoDB shell
mongo

// Switch to the prospectis database
use prospectis

// Create indexes for the research_papers collection
db.research_papers.createIndex({ "paper_id": 1 }, { unique: true })
db.research_papers.createIndex({ "source": 1, "source_id": 1 }, { unique: true })
db.research_papers.createIndex({ "published_date": -1 })
db.research_papers.createIndex({ "title": "text", "abstract": "text" })
db.research_papers.createIndex({ "categories": 1 })
db.research_papers.createIndex({ "authors.name": 1 })
db.research_papers.createIndex({ "ingestion_date": -1 })
```

### Recommended Hardware Requirements

For small to medium deployments:
- 2+ CPU cores
- 4+ GB RAM
- 50+ GB storage (SSD recommended)

For larger deployments, scale resources accordingly based on data volume and query patterns.

## Backup and Restoration

### Creating a Backup

```bash
# Local backup using mongodump
mongodump --db prospectis --out /path/to/backup/directory

# For Atlas, use Atlas UI or mongodump with connection string
mongodump --uri "mongodb+srv://<username>:<password>@<cluster-url>/prospectis" --out /path/to/backup/directory
```

### Restoring from Backup

```bash
# Local restoration using mongorestore
mongorestore --db prospectis /path/to/backup/directory/prospectis

# For Atlas, use Atlas UI or mongorestore with connection string
mongorestore --uri "mongodb+srv://<username>:<password>@<cluster-url>/prospectis" --db prospectis /path/to/backup/directory/prospectis
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure MongoDB service is running and listening on the expected port
   ```bash
   sudo systemctl status mongod
   ```

2. **Authentication Failure**: Verify username, password, and database name in connection string

3. **Insufficient Disk Space**: Check available storage
   ```bash
   df -h
   ```

4. **Slow Queries**: Ensure indexes are created and properly utilized

For more assistance, consult the [MongoDB Documentation](https://docs.mongodb.com/) or create an issue in the project repository.