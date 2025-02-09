# Cache Manager module.Provides a CacheManager class for atomic file writes and automatic cache cleaning.


import os
import time
from contextlib import contextmanager
import filelock

DEFAULT_CACHE_DIR = "./cache"  # Default directory for cache files

class CacheManager:
    """
    Manages cache files with atomic write operations and automatic cleaning of outdated cache files.
    """
    
    def __init__(self, cache_dir=None):
        """
        Initialize the CacheManager with a specified cache directory.
        If no directory is provided, the default cache directory is used.
        
        Args:
            cache_dir (str, optional): The directory to store cache files.
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @contextmanager
    def atomic_write(self, path):
        """
        Provide an atomic write context manager.
        
        This context manager yields a temporary file path for writing.
        Upon exiting the context, the temporary file is renamed to the target path.
        
        Args:
            path (str): The target file path.
        
        Yields:
            str: A temporary file path for writing.
        """
        lock_path = path + ".lock"
        with filelock.FileLock(lock_path):
            temp_path = path + ".tmp"
            yield temp_path
            os.rename(temp_path, path)
    
    def clear_old_cache(self, max_age_days=30):
        """
        Automatically clear cache files older than the specified number of days.
        
        Args:
            max_age_days (int): The maximum age (in days) for cache files to keep.
                                Files older than this will be removed.
        """
        now = time.time()
        for fname in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, fname)
            if os.stat(path).st_mtime < now - max_age_days * 86400:
                os.remove(path)
