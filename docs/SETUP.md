# 🚀 Detailed Setup Guide

This guide provides comprehensive step-by-step instructions for setting up the Book Recommendation System on your local machine.

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Python 3.8 or higher** installed
- [ ] **Git** for cloning the repository
- [ ] **OpenAI API Key** (sign up at https://platform.openai.com/)
- [ ] **Internet connection** for downloading dependencies and data
- [ ] **At least 4GB RAM** for processing embeddings
- [ ] **2GB free disk space** for Neo4j and data files

## 🔧 Step-by-Step Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/BookRecommender.git
cd BookRecommender

# Verify you're in the correct directory
ls -la
# You should see files like streamlit_app.py, merge_all_collections.py, etc.
```

### Step 2: Set Up Python Environment

#### Option A: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv book_recommender_env

# Activate virtual environment
# On Windows:
book_recommender_env\Scripts\activate
# On macOS/Linux:
source book_recommender_env/bin/activate

# Verify activation (you should see the environment name in your prompt)
```

#### Option B: Using Conda
```bash
# Create conda environment
conda create -n book_recommender python=3.9
conda activate book_recommender
```

### Step 3: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | grep streamlit
pip list | grep neo4j
pip list | grep openai
```

### Step 4: Set Up Neo4j Database

#### Option A: Neo4j Aura Cloud (Recommended)
1. **Create a free Neo4j Aura account** at https://neo4j.com/cloud/aura/
2. **Create a new database instance** (free tier available)
3. **Save your connection details**:
   - Connection URI (e.g., `neo4j+s://xxxxx.databases.neo4j.io`)
   - Username (usually `neo4j`)
   - Password (set during creation)

#### Option B: Local Neo4j Installation
If you prefer to run Neo4j locally:
1. **Download Neo4j Desktop** from https://neo4j.com/download/
2. **Install and create a new database**
3. **Start the database** and note the connection details
4. **Default connection**: `bolt://localhost:7687`

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env  # macOS/Linux
# Or create manually in Windows
```

Add the following content to `.env`:

```env
# Neo4j Configuration (Update with your Neo4j Aura or local details)
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io  # For Aura Cloud
# NEO4J_URI=bolt://localhost:7687  # For local installation
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_DATABASE=neo4j

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Ollama Configuration (if using local models)
OLLAMA_BASE_URL=http://localhost:11434
```

**Important**: 
- Replace `your_openai_api_key_here` with your actual OpenAI API key
- Replace `your_neo4j_password_here` with your Neo4j password
- Update the `NEO4J_URI` with your actual Neo4j connection details

### Step 6: Test Database Connection

```bash
# Test Neo4j connection
python tests/test_neo4j_connection.py

# Expected output:
# ✅ Neo4j connection successful
# Database info: {'name': 'neo4j', 'creation_time': '...'}
```

**Note**: If using Neo4j Aura, you don't need to change the default password as it's set during database creation.

### Step 7: Test OpenAI Connection

```bash
python -c "
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
print('OpenAI API key configured successfully')
"
```

## 📊 Data Setup and Processing

### Step 8: Process Existing Data (If Available)

If you have the collected data files:

```bash
# Generate embeddings for all books
python scripts/processing/vectorize_books.py

# Expected output:
# Processing 2131 books...
# Generated embeddings for 2131 books
# Saved to books_with_embeddings.json

# Upload data to Neo4j
python scripts/database/upload_to_neo4j.py

# Expected output:
# Uploaded 2118 books to Neo4j
# Created 541 publishers
# Created 2011 relationships
```

### Step 9: Collect New Data (Optional)

To collect fresh book data:

```bash
# Collect books by genre (this may take time)
python scripts/collection/collect_horror_books.py
python scripts/collection/collect_mystery_books.py
python scripts/collection/collect_romance_books.py
# ... run other collectors as needed

# Merge all collected data
python scripts/processing/merge_all_collections.py

# Generate embeddings and upload
python scripts/processing/vectorize_books.py
python scripts/database/upload_to_neo4j.py
```

## 🌐 Launch the Application

### Step 10: Start the Web Interface

```bash
# Start Streamlit app
streamlit run app/streamlit_app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501
```

### Step 11: Verify Everything Works

1. **Open your browser** and navigate to `http://localhost:8501`
2. **Check the connection status** - you should see green checkmarks for:
   - ✅ Connected to Neo4j database
   - ✅ Loaded X books with embeddings
3. **Test search functionality** - search for a book title
4. **Test recommendations** - click "Find Similar Books" on any result

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "Neo4j connection failed"
```bash
# Check if Neo4j is running
# Look for Neo4j process in Task Manager (Windows) or Activity Monitor (macOS)

# Restart Neo4j
# Stop: Ctrl+C in the Neo4j console
# Start: .\neo4j-community-5.26.0\bin\neo4j.bat console

# Check firewall settings - ensure port 7687 is not blocked
```

#### Issue: "OpenAI API key not found"
```bash
# Verify .env file exists and contains your API key
cat .env  # macOS/Linux
type .env # Windows

# Check API key format (should start with 'sk-')
# Verify no extra spaces or quotes around the key
```

#### Issue: "Java not found" when starting Neo4j
```bash
# Verify Java installation
java -version

# Set JAVA_HOME if not set
export JAVA_HOME=/path/to/jdk-11.0.2  # macOS/Linux
$env:JAVA_HOME = "C:\path\to\jdk-11.0.2"  # Windows PowerShell

# Add to PATH
export PATH=$JAVA_HOME/bin:$PATH  # macOS/Linux
$env:PATH += ";$env:JAVA_HOME\bin"  # Windows PowerShell
```

#### Issue: "Module not found" errors
```bash
# Ensure virtual environment is activated
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.8+
```

#### Issue: Streamlit app loads but shows errors
```bash
# Check all services are running:
# 1. Neo4j database (should show "Started." in console)
# 2. Check data files exist:
ls books_with_embeddings.json
ls merged_book_dataset.json

# If files missing, run data processing:
python vectorize_books.py
python upload_to_neo4j.py
```

### Performance Optimization

#### For Large Datasets
```bash
# Increase memory allocation for Python
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4

# For Neo4j performance tuning, edit:
# neo4j-community-5.26.0/conf/neo4j.conf
# Uncomment and adjust:
# server.memory.heap.initial_size=1G
# server.memory.heap.max_size=2G
```

#### For Slow Embedding Generation
```bash
# Process in smaller batches
# Edit vectorize_books.py and reduce batch_size from 100 to 50

# Monitor OpenAI API rate limits
# The script includes automatic rate limiting
```

## 🔄 Maintenance and Updates

### Regular Maintenance Tasks

#### Update Book Data
```bash
# Monthly data refresh
python collect_new_releases.py
python merge_all_collections.py
python vectorize_books.py
python upload_to_neo4j.py
```

#### Database Cleanup
```bash
# Remove duplicate entries
python -c "
from neo4j import GraphDatabase
# Connect and run cleanup queries
# See neo4j_schema_and_queries.py for examples
"
```

#### Backup Data
```bash
# Backup JSON files
cp *.json backup/

# Backup Neo4j database
# Stop Neo4j first, then copy data directory
cp -r neo4j-community-5.26.0/data backup/neo4j-data
```

### Updating Dependencies
```bash
# Update Python packages
pip list --outdated
pip install --upgrade streamlit neo4j openai

# Update requirements.txt
pip freeze > requirements.txt
```

## 📱 Advanced Configuration

### Custom Neo4j Configuration
Edit `neo4j-community-5.26.0/conf/neo4j.conf`:

```conf
# Enable additional connectors
server.bolt.listen_address=0.0.0.0:7687
server.http.listen_address=0.0.0.0:7474

# Memory settings
server.memory.heap.initial_size=1G
server.memory.heap.max_size=2G
server.memory.pagecache.size=512M

# Security settings
dbms.security.auth_enabled=true
```

### Streamlit Configuration
Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🚀 Production Deployment

### Docker Setup (Advanced)
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

### Environment-Specific Configurations
```bash
# Development
export ENVIRONMENT=development
export DEBUG=true

# Production
export ENVIRONMENT=production
export DEBUG=false
export NEO4J_URI=bolt://production-neo4j:7687
```

## 📞 Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs** - Look for error messages in the console
2. **Review the README.md** - Additional context and examples
3. **Search existing issues** - GitHub Issues tab
4. **Create a new issue** - Include error messages and system info
5. **Join the community** - Discord/Slack channels (if available)

### System Information for Bug Reports
```bash
# Gather system info for bug reports
python --version
pip list | grep -E "(streamlit|neo4j|openai)"
java -version
# Include OS version and any error messages
```

---

**Setup Complete! 🎉**

You should now have a fully functional Book Recommendation System running locally. Visit `http://localhost:8501` to start exploring book recommendations!