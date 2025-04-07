# Deployment Guide

This document provides instructions for deploying the Prospectis research pipeline in various environments.

## Local Deployment

### Prerequisites

- Python 3.8 or higher
- MongoDB installed and running
- API keys for IEEE Xplore and Semantic Scholar (optional but recommended)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/prospectis.git
   cd prospectis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your settings
   ```

5. Run the pipeline manually to test:
   ```bash
   python scripts/run_pipeline.py
   ```

6. Set up a scheduled job:
   ```bash
   # Using the Python scheduler
   python scripts/scheduler.py
   
   # Alternative: Using cron (Linux/macOS)
   crontab -e
   # Add: 0 0 * * * cd /path/to/prospectis && /path/to/python /path/to/prospectis/scripts/run_pipeline.py
   ```

## Docker Deployment

### Prerequisites

- Docker installed
- Docker Compose installed (optional, for multi-container setup)

### Steps

1. Build the Docker image:
   ```bash
   docker build -t prospectis-pipeline .
   ```

2. Run the container:
   ```bash
   docker run -d \
     --name prospectis-pipeline \
     -e MONGODB_URI=mongodb://mongo:27017 \
     -e MONGODB_DB=prospectis \
     -e IEEE_API_KEY=your_ieee_api_key \
     -e SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key \
     prospectis-pipeline
   ```

3. For a complete setup with MongoDB, create a `docker-compose.yml` file:
   ```yaml
   version: '3'
   
   services:
     mongo:
       image: mongo:latest
       restart: always
       volumes:
         - mongodb_data:/data/db
       ports:
         - "27017:27017"
   
     pipeline:
       build: .
       restart: always
       depends_on:
         - mongo
       environment:
         - MONGODB_URI=mongodb://mongo:27017
         - MONGODB_DB=prospectis
         - IEEE_API_KEY=${IEEE_API_KEY}
         - SEMANTIC_SCHOLAR_API_KEY=${SEMANTIC_SCHOLAR_API_KEY}
         - SCHEDULER_INTERVAL_HOURS=24
   
   volumes:
     mongodb_data:
   ```

4. Run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

## Cloud Deployment

### AWS Deployment

#### Using EC2

1. Launch an EC2 instance with appropriate resources
2. Install Docker on the instance
3. Deploy using the Docker instructions above
4. Set up CloudWatch to monitor the instance and application

#### Using ECS/Fargate

1. Create an ECS cluster
2. Define a task definition with the container configuration
3. Create a service to run the task
4. Set up a scheduled task using EventBridge (CloudWatch Events)

### GCP Deployment

#### Using Compute Engine

1. Launch a VM instance
2. Install Docker on the instance
3. Deploy using the Docker instructions above
4. Set up Cloud Monitoring

#### Using Cloud Run Jobs

1. Build and push the container to Container Registry or Artifact Registry
2. Create a Cloud Run Job with the container
3. Set up Cloud Scheduler to trigger the job at regular intervals

### Azure Deployment

#### Using Virtual Machines

1. Create an Azure VM
2. Install Docker on the VM
3. Deploy using the Docker instructions above
4. Set up Azure Monitor

#### Using Container Instances

1. Build and push the container to Azure Container Registry
2. Create a Container Instance with the container
3. Set up Azure Logic Apps or Azure Functions to trigger the job at regular intervals

## Serverless Deployment

For a serverless approach, consider separating the pipeline into smaller functions:

1. **AWS Lambda / GCP Cloud Functions / Azure Functions**:
   - Create a function to fetch papers from each source
   - Create a function to process and store papers
   - Set up triggers to run at scheduled intervals

2. **Managed Database Services**:
   - Use MongoDB Atlas for the database
   - Configure appropriate security settings for access

## Production Considerations

### Logging and Monitoring

1. **Log Management**:
   - Use a centralized logging service (CloudWatch, StackDriver, Azure Monitor)
   - Set up log alerts for errors and warnings

2. **Application Monitoring**:
   - Monitor pipeline execution time
   - Track number of papers fetched/stored
   - Set up alerts for pipeline failures

### Security

1. **API Keys and Secrets**:
   - Use secrets management services (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
   - Never store secrets in code or Docker images

2. **Network Security**:
   - Restrict MongoDB access to application IPs only
   - Use VPC/private networking when possible

### Scaling

1. **Database Scaling**:
   - Start with a small MongoDB instance and scale as needed
   - Consider sharding for very large datasets

2. **Pipeline Scaling**:
   - Implement parallel processing for multiple sources
   - Use connection pooling for database connections

### Backup and Recovery

1. **Regular Backups**:
   - Set up automated backups for the MongoDB database
   - Store backups in a separate storage location

2. **Disaster Recovery**:
   - Document recovery procedures
   - Test restoration process periodically

## Continuous Integration / Continuous Deployment

1. **CI/CD Pipeline**:
   - Use GitHub Actions, GitLab CI, Jenkins, or other CI/CD tools
   - Automate testing, building, and deployment

2. **Testing**:
   - Run unit tests before deployment
   - Perform integration tests with test databases

3. **Deployment Strategies**:
   - Use blue/green deployment for zero-downtime updates
   - Implement rollback procedures for failed deployments

## Cost Optimization

1. **Resource Management**:
   - Right-size compute resources for the workload
   - Use spot/preemptible instances for non-critical workloads

2. **Scheduling**:
   - Run pipelines during off-peak hours if possible
   - Optimize pipeline frequency based on data refresh needs