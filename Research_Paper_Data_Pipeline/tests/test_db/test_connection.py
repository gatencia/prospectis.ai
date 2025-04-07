"""
Tests for MongoDB connection handling.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from research_pipeline.db.connection import (
    get_mongo_client,
    get_db,
    get_papers_collection,
    close_connection
)


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing."""
    os.environ["MONGODB_URI"] = "mongodb://testserver:27017"
    os.environ["MONGODB_DB"] = "test_db"
    os.environ["MONGODB_COLLECTION_PAPERS"] = "test_papers"
    yield
    # Clean up
    os.environ.pop("MONGODB_URI", None)
    os.environ.pop("MONGODB_DB", None)
    os.environ.pop("MONGODB_COLLECTION_PAPERS", None)


@patch('research_pipeline.db.connection.MongoClient')
def test_get_mongo_client(mock_mongo_client, mock_env_vars):
    """Test getting MongoDB client."""
    # Setup mock client
    mock_instance = MagicMock()
    mock_mongo_client.return_value = mock_instance
    
    # Call get_mongo_client
    client = get_mongo_client()
    
    # Assertions
    assert client == mock_instance
    mock_mongo_client.assert_called_once_with("mongodb://testserver:27017")
    
    # Call again to test singleton behavior
    client2 = get_mongo_client()
    assert client2 == mock_instance
    # Should still be called only once
    mock_mongo_client.assert_called_once()


@patch('research_pipeline.db.connection.get_mongo_client')
def test_get_db(mock_get_client, mock_env_vars):
    """Test getting MongoDB database."""
    # Setup mock client and db
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.__getitem__.return_value = mock_db
    mock_get_client.return_value = mock_client
    
    # Call get_db
    db = get_db()
    
    # Assertions
    assert db == mock_db
    mock_get_client.assert_called_once()
    mock_client.__getitem__.assert_called_once_with("test_db")


@patch('research_pipeline.db.connection.get_db')
def test_get_papers_collection(mock_get_db, mock_env_vars):
    """Test getting papers collection."""
    # Setup mock db and collection
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_get_db.return_value = mock_db
    
    # Call get_papers_collection
    collection = get_papers_collection()
    
    # Assertions
    assert collection == mock_collection
    mock_get_db.assert_called_once()
    mock_db.__getitem__.assert_called_once_with("test_papers")


@patch('research_pipeline.db.connection._mongo_client')
def test_close_connection(mock_mongo_client):
    """Test closing MongoDB connection."""
    # Setup mock client
    mock_client_instance = MagicMock()
    mock_mongo_client.return_value = mock_client_instance
    
    # Set the global variable
    from research_pipeline.db.connection import _mongo_client
    _mongo_client = mock_client_instance
    
    # Call close_connection
    close_connection()
    
    # Assertions
    mock_client_instance.close.assert_called_once()
    assert _mongo_client is None