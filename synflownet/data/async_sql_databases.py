import sqlite3
import time
from abc import ABC
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class AsyncSQLDatabase(ABC):
    """Base class for SQLite databases with concurrent access handling"""

    def __init__(self, db_path: str, max_size: int = 100_000, max_batch_size: int = 1000):
        self.db_path = Path(db_path)
        self.max_size = max_size
        self.max_batch_size = max_batch_size
        self._initialize_db()

    def _initialize_db(self):
        """Initialize database or connect to existing one"""
        with self._get_db() as (conn, cursor):
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entries'")
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                logger.info(f"Creating new table in database at {self.db_path}")
                cursor.execute(
                    """
                    CREATE TABLE entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        smile TEXT,
                        traj BLOB,
                        reward FLOAT,
                        timestamp FLOAT,
                        info TEXT
                    )
                """
                )
            else:
                logger.info(f"Connected to existing database at {self.db_path}")
                cursor.execute("PRAGMA table_info(entries)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                expected_schema = {
                    "id": "INTEGER",
                    "smile": "TEXT",
                    "traj": "BLOB",
                    "reward": "FLOAT",
                    "timestamp": "FLOAT",
                    "info": "TEXT",
                }
                if columns != expected_schema:
                    raise ValueError(f"Database schema mismatch. Expected {expected_schema}, got {columns}")
            logger.info(
                f"DB info: current_size={self.get_db_size(fast=False):,},"
                f" max_size={self.max_size:,}, max_batch_size={self.max_batch_size:,}"
            )

    @contextmanager
    def _get_db(self):
        """Get database connection"""

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Attempt to connect to the database
        max_attempts = 5
        attempt = 0
        while True:
            try:
                conn = sqlite3.connect(self.db_path, timeout=300.0)
                cursor = conn.cursor()
                yield conn, cursor
                conn.commit()
                break
            except sqlite3.OperationalError as e:
                attempt += 1
                logger.warning(f"Database error: {str(e)}")
                logger.warning(f"Database path: {self.db_path}")
                if attempt >= max_attempts:
                    logger.error(f"Failed to connect to database after {max_attempts} attempts")
                    raise
                logger.warning(f"Database locked, retrying... (attempt {attempt}/{max_attempts})")
                time.sleep(0.1 * attempt)
            finally:
                if "conn" in locals():
                    conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics"""
        with self._get_db() as (conn, cursor):
            cursor.execute(
                """
                SELECT
                    COUNT(*) as count,
                    AVG(reward) as mean_reward,
                    MIN(reward) as min_reward,
                    MAX(reward) as max_reward,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM entries
            """
            )
            count, mean, min_, max_, oldest, newest = cursor.fetchone()
            return {
                "size": count,
                "mean_reward": mean if mean is not None else None,
                "min_reward": min_,
                "max_reward": max_,
                "oldest_entry_age": time.time() - oldest if oldest else None,
                "newest_entry_age": time.time() - newest if newest else None,
            }

    def clear(self):
        """Clear all entries from database"""
        with self._get_db() as (conn, cursor):
            cursor.execute("DELETE FROM entries")

    def peek(self, batch_size: int) -> tuple[list[float], list[str], list[bytes], list[float], list[float]]:
        """View entries without removing them, ordered by timestamp"""
        with self._get_db() as (conn, cursor):
            cursor.execute(
                "SELECT id, smile, traj, reward, timestamp, info FROM entries ORDER BY timestamp LIMIT ?",
                (batch_size,),
            )
            results = cursor.fetchall()
            data = zip(*results)
            return list(data)

    def get_db_size(self, fast: bool = False) -> int:
        """Get current number of entries in the database"""
        with self._get_db() as (conn, cursor):
            if fast:
                result = cursor.execute("SELECT seq FROM sqlite_sequence WHERE name='entries'").fetchone()
                return result[0] if result is not None else 0
            else:
                return cursor.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def check_entries_for_errors(self, smiles: list[str], trajs: list[bytes]):
        """Validate input data before insertion"""
        if len(trajs) > self.max_batch_size:
            raise ValueError(f"Batch size must be less than {self.max_batch_size}")
        if not smiles or not trajs:
            raise ValueError("Empty inputs not allowed")
        if not all(isinstance(s, str) for s in smiles):
            raise ValueError("All smiles must be strings")
        if not all(isinstance(t, bytes) for t in trajs):
            raise ValueError("All trajs must be bytes (pickle-serialized)")
        if len(smiles) != len(trajs):
            raise ValueError("Length of smiles and trajs must match")


class PersistentReplayBuffer(AsyncSQLDatabase):
    """
    A persistent buffer for storing rewarded trajectories

    Here, we assume that several reward processes are running in parallel, and each process is pushing
    trajectories to the buffer as they are computed. Our synflownet program will use this replay buffer
    to sample trajectories for training. We call it `Persistent` since it is designed to be conserved
    across training runs (unless the reward processes change e.g. different model version or target protein).
    The decision to load an existing buffer or create a new one is fully handled by the db_path parameter.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Checks if data has previously been removed from the database to warn the user
        # as this would cause get_db_size(fast=True) != get_db_size(fast=False)
        # Note that for persistent replay buffer, data generally should not have been removed
        # i.e. auto-increment largest value should be equal to the number of rows in the table.
        if self.get_db_size(fast=True) != self.get_db_size(fast=False):
            logger.warning(
                "Warning: data has previously been removed from the database. "
                "This will cause get_db_size(fast=True) != get_db_size(fast=False)"
            )

    def check_entries_for_errors(
        self,
        smiles: list[str],
        trajs: list[bytes],
        rewards: Optional[list[float]] = None,
        infos: Optional[list[str]] = None,
    ):
        super().check_entries_for_errors(smiles, trajs)
        if not rewards or len(smiles) != len(rewards):
            raise ValueError("Length of smiles, trajs, and rewards must match")
        if not all(isinstance(r, (int, float)) for r in rewards):
            raise ValueError("All rewards must be floats")
        if infos is not None and len(smiles) != len(infos):
            raise ValueError("Length of smiles and infos must match")
        if infos is not None and not all(isinstance(i, str) for i in infos):
            raise ValueError("All infos must be strings")

    def push(self, smiles: list[str], trajs: list[bytes], rewards: list[float], infos: Optional[list[str]] = None):
        """
        Add rewarded trajectories to the buffer.
        When buffer is full, removes oldest entries to make space (FIFO behavior).
        """
        self.check_entries_for_errors(smiles, trajs, rewards, infos)
        current_size = self.get_db_size()

        with self._get_db() as (conn, cursor):
            cursor.execute("BEGIN IMMEDIATE")

            # If we need to make space, remove oldest entries
            if current_size + len(smiles) > self.max_size:
                entries_to_remove = current_size + len(smiles) - self.max_size
                cursor.execute(
                    """
                    DELETE FROM entries
                    WHERE id IN (
                        SELECT id FROM entries
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                """,
                    (entries_to_remove,),
                )

            # Insert new entries
            timestamps = [time.time()] * len(smiles)
            if infos is None:
                infos = [""] * len(smiles)
            cursor.executemany(
                "INSERT INTO entries (smile, traj, reward, timestamp, info) VALUES (?, ?, ?, ?, ?)",
                zip(smiles, trajs, rewards, timestamps, infos),
            )

    def sample(self, batch_size: int, min_reward: Optional[float] = None) -> list[tuple[int, str, bytes, float, float, str]]:
        """
        Sample trajectories without replacement, optionally filtering by minimum reward.
        Returns: List of tuples (id, smile, traj, reward, timestamp, info)
        """
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size must be less than {self.max_batch_size}")

        with self._get_db() as (conn, cursor):
            query = "SELECT id, smile, traj, reward, timestamp, info FROM entries"
            params = []

            if min_reward is not None:
                query += " WHERE reward >= ?"
                params.append(min_reward)

            query += " ORDER BY RANDOM() LIMIT ?"
            params.append(batch_size)

            cursor.execute(query, tuple(params))
            data = list(zip(*cursor.fetchall()))

            if not data or len(data[0]) < batch_size:
                raise ValueError(f"Not enough entries in database. Requested {batch_size} but only got {len(data[0]) if data else 0}")

            return data

    def get_last_n_entries(self, batch_size: int) -> list[tuple[int, str, bytes, float, float, str]]:
        """
        Get the last n entries from the database
        Returns: List of tuples (id, smile, traj, reward, timestamp, info)
        """
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size must be less than {self.max_batch_size}")

        with self._get_db() as (conn, cursor):
            query = "SELECT id, smile, traj, reward, timestamp, info FROM entries ORDER BY timestamp DESC LIMIT ?"
            cursor.execute(query, (batch_size,))
            data = list(zip(*cursor.fetchall()))

            if not data or len(data[0]) < batch_size:
                raise ValueError(f"Not enough entries in database. Requested {batch_size} but only got {len(data[0]) if data else 0}")

            return data


class RewardQueue(AsyncSQLDatabase):
    """
    A queue for trajectories awaiting reward computation.

    A reward process is any program running in parallel, which synflownet does not control and does not
    need to be aware of. This RewardQueue structure is meant to accomodate long-running reward computations,
    with an arbitrary number of such reward processes concurrently sampling without replacement trajectories
    from the queue. This is a one-way queue, once a trajectory is popped from the queue by a reward process,
    it is not returned to the queue.

    We implement this as a sqlite database, which is a simple and robust way to support concurrent access.
    """

    def push(self, smiles: list[str], trajs: list[bytes], allow_fifo: bool = False):
        """Push new trajectories to the queue

        Args:
            smiles: List of SMILES strings
            trajs: List of pickled trajectory bytes
            allow_fifo: If True, remove oldest entries when full (FIFO behavior).
                       If False, raise ValueError when full (halt behavior).
        """
        current_size = self.get_db_size()

        if current_size >= self.max_size and not allow_fifo:
            raise ValueError("Cannot push. Queue is full")

        self.check_entries_for_errors(smiles, trajs)
        rewards = [None] * len(smiles)
        timestamps = [time.time()] * len(smiles)
        infos = [""] * len(smiles)

        with self._get_db() as (conn, cursor):
            cursor.execute("BEGIN IMMEDIATE")  # Blocks other writers AND readers

            # If we need to make space and FIFO is allowed, remove oldest entries
            if allow_fifo and ((current_size + len(smiles)) > self.max_size):
                entries_to_remove = current_size + len(smiles) - self.max_size
                cursor.execute(
                    """
                    DELETE FROM entries
                    WHERE id IN (
                        SELECT id FROM entries
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                """,
                    (entries_to_remove,),
                )

            cursor.executemany(
                "INSERT INTO entries (smile, traj, reward, timestamp, info) VALUES (?, ?, ?, ?, ?)",
                zip(smiles, trajs, rewards, timestamps, infos),
            )

    def pop(self, batch_size: int) -> tuple[list[int], list[str], list[bytes], list[float], list[float], list[str]]:
        """Pop a batch of entries from the queue (no replacement)"""
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size must be less than {self.max_batch_size}")

        with self._get_db() as (conn, cursor):
            cursor.execute("BEGIN IMMEDIATE")  # Blocks other writers AND readers
            cursor.execute(
                "SELECT id, smile, traj, reward, timestamp, info FROM entries ORDER BY timestamp DESC LIMIT ?",
                (batch_size,),
            )
            entries = cursor.fetchall()
            data = zip(*entries)

            if not entries:
                return []
            else:  # Only attempt delete if we found entries
                ids_to_remove = [str(id) for id, _, _, _, _, _ in entries]
                cursor.execute(f"DELETE FROM entries WHERE id IN ({','.join(ids_to_remove)})")
            return list(data)


class RewardCache:
    """
    A persistent cache for storing and retrieving molecule rewards.

    This class purposedly does not save trajectories, only SMILES and rewards.
    Multiple trajectories could lead to the same terminal states and should be
    enforced separately by the model in the PersistentReplayBuffer.
    However, by caching on terminal states we only make the assumption that the reward
    is deterministic and will be the same for the same terminal state (not sctrictly
    true for Boltz-2, but this becomes a compute-vs-precision trade-off).

    This class provides thread-safe caching of SMILES -> reward mappings
    using SQLite with WAL mode for concurrent access.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database, creating tables and ensuring integrity."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._is_valid_sqlite():
            raise ValueError(f"Corrupted or invalid DB at {self.db_path}. Please delete it and try again.")

        with self._connect() as (conn, cursor):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    smiles TEXT NOT NULL,
                    reward REAL NOT NULL,
                    info TEXT,
                    UNIQUE(smiles)
                )
            """)

    def _is_valid_sqlite(self) -> bool:
        """Check if the database file is a valid SQLite database."""
        if not self.db_path.exists():
            return True
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA schema_version")
                conn.execute("PRAGMA integrity_check(1)")
                conn.execute("SELECT COUNT(*) FROM sqlite_master")
            return True
        except sqlite3.DatabaseError as e:
            logger.warning(f"Database validation failed: {repr(e)}")
            return False

    @contextmanager
    def _connect(self):
        """Robust DB connection context manager with retry and safe cleanup."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        max_attempts = 5
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("PRAGMA busy_timeout=5000")
                cursor = conn.cursor()

                try:
                    yield conn, cursor
                    conn.commit()
                    return  # Successful completion
                except Exception:
                    # If an exception occurs during yield, rollback and re-raise
                    try:
                        conn.rollback()
                    except Exception:
                        pass  # Ignore rollback errors
                    raise
                finally:
                    # Always close the connection
                    try:
                        conn.close()
                    except Exception:
                        pass  # Ignore close errors


            except sqlite3.OperationalError as e:
                last_exception = e
                if "database is locked" in str(e).lower() or "disk i/o error" in str(e).lower():
                    logger.warning(f"Cache DB error (attempt {attempt}/{max_attempts}): {repr(e)}")
                    if attempt < max_attempts:
                        time.sleep(0.1 * attempt * (1 + 0.1 * attempt))
                        continue
                # If it's not a retryable error or we've exhausted attempts, re-raise
                raise
            except Exception:
                # For non-operational errors, don't retry
                raise

        # If we get here, we've exhausted all attempts
        if last_exception:
            raise last_exception
        else:
            raise sqlite3.OperationalError("Failed to connect to database after all attempts")

    def get_hits(self, smiles_list: list[str]) -> list[tuple[str, float, str]]:
        """
        Retrieve cached rewards for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings to look up

        Returns:
            List of tuples (smiles, reward, info) for found entries

        Raises:
            ValueError: If smiles_list is empty
            sqlite3.Error: If database operation fails
        """
        if not smiles_list:
            return []

        if not all(isinstance(s, str) for s in smiles_list):
            raise ValueError("All items in smiles_list must be strings")

        placeholders = ",".join("?" for _ in smiles_list)
        query = f"SELECT smiles, reward, info FROM entries WHERE smiles IN ({placeholders})"

        with self._connect() as (conn, cursor):
            cursor.execute(query, smiles_list)
            return cursor.fetchall()

    def get_db_size(self, fast: bool = False) -> int:
        """Get current number of entries in the database"""
        with self._connect() as (conn, cursor):
            cursor.execute("BEGIN IMMEDIATE")
            if fast:
                result = cursor.execute("SELECT seq FROM sqlite_sequence WHERE name='entries'").fetchone()
                return result[0] if result is not None else 0
            else:
                return cursor.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def insert_entries(self, entries: list[tuple[str, float, str]]):
        """
        Insert new entries into the cache.

        Args:
            entries: List of tuples (smiles, reward, info) to insert

        Raises:
            ValueError: If entries format is invalid
            sqlite3.Error: If database operation fails
        """
        if not entries:
            return

        # Validate entry format
        for i, entry in enumerate(entries):
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError(
                    f"Entry {i} must be a tuple of length 3, got {type(entry)} "
                    f"with length {len(entry) if hasattr(entry, '__len__') else 'unknown'}"
                )

            smiles, reward, info = entry
            if not isinstance(smiles, str):
                raise ValueError(f"Entry {i}: smiles must be string, got {type(smiles)}")
            if not isinstance(reward, (int, float)):
                raise ValueError(f"Entry {i}: reward must be numeric, got {type(reward)}")
            if not isinstance(info, str):
                raise ValueError(f"Entry {i}: info must be string, got {type(info)}")

        with self._connect() as (conn, cursor):
            cursor.execute("BEGIN IMMEDIATE")
            cursor.executemany(
                "INSERT OR IGNORE INTO entries (smiles, reward, info) VALUES (?, ?, ?)",
                entries,
            )
