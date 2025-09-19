# redis_config.py - Redis Cloud Configuration
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, Optional

# Load environment variables
load_dotenv()

@dataclass
class RedisConfig:
    """Redis Cloud configuration settings"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    username: Optional[str] = None
    db: int = 0
    ssl: bool = False
    ssl_cert_reqs: str = "none"
    decode_responses: bool = True
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    @classmethod
    def from_env(cls):
        """Create Redis config from environment variables"""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            username=os.getenv("REDIS_USERNAME"),
            db=int(os.getenv("REDIS_DB", "0")),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            ssl_cert_reqs=os.getenv("REDIS_SSL_CERT_REQS", "none"),
            decode_responses=True,
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "30")),
            socket_connect_timeout=int(os.getenv("REDIS_CONNECT_TIMEOUT", "30")),
            retry_on_timeout=True,
            health_check_interval=int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for redis.Redis()"""
        config = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'decode_responses': self.decode_responses,
            'socket_timeout': self.socket_timeout,
            'socket_connect_timeout': self.socket_connect_timeout,
            'retry_on_timeout': self.retry_on_timeout,
            'health_check_interval': self.health_check_interval,
        }
        
        if self.password:
            config['password'] = self.password
        if self.username:
            config['username'] = self.username
        if self.ssl:
            config['ssl'] = True
            config['ssl_cert_reqs'] = self.ssl_cert_reqs
            
        return config

@dataclass
class AgentConfig:
    """Agent configuration settings"""
    batch_size: int = 1000
    num_batches: int = 5
    fraud_rate: float = 0.05
    high_risk_rate: float = 0.1
    log_level: str = "INFO"
    num_customers: int = 100
    returning_customer_rate: float = 0.7
    
    # Redis-specific settings
    key_prefix: str = "financial"
    transaction_ttl: int = 2592000  # 30 days in seconds
    use_pipeline: bool = True
    pipeline_size: int = 100
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls(
            batch_size=int(os.getenv("BATCH_SIZE", "1000")),
            num_batches=int(os.getenv("NUM_BATCHES", "5")),
            fraud_rate=float(os.getenv("FRAUD_RATE", "0.05")),
            high_risk_rate=float(os.getenv("HIGH_RISK_RATE", "0.1")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            num_customers=int(os.getenv("NUM_CUSTOMERS", "100")),
            returning_customer_rate=float(os.getenv("RETURNING_CUSTOMER_RATE", "0.7")),
            key_prefix=os.getenv("REDIS_KEY_PREFIX", "financial"),
            transaction_ttl=int(os.getenv("TRANSACTION_TTL", "2592000")),
            use_pipeline=os.getenv("USE_REDIS_PIPELINE", "true").lower() == "true",
            pipeline_size=int(os.getenv("PIPELINE_SIZE", "100"))
        )

# Redis Cloud .env template
ENV_TEMPLATE = """# Redis Cloud Configuration
# Get these from your Redis Cloud dashboard
REDIS_HOST=your-redis-cloud-endpoint.com
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_USERNAME=default
REDIS_SSL=true
REDIS_SSL_CERT_REQS=none
REDIS_DB=0

# Connection settings
REDIS_SOCKET_TIMEOUT=30
REDIS_CONNECT_TIMEOUT=30
REDIS_HEALTH_CHECK_INTERVAL=30

# Agent Configuration
BATCH_SIZE=1000
NUM_BATCHES=5
FRAUD_RATE=0.05
HIGH_RISK_RATE=0.1
LOG_LEVEL=INFO
NUM_CUSTOMERS=100
RETURNING_CUSTOMER_RATE=0.7

# Redis-specific settings
REDIS_KEY_PREFIX=financial
TRANSACTION_TTL=2592000
USE_REDIS_PIPELINE=true
PIPELINE_SIZE=100
"""

def create_env_file():
    """Create a sample .env file for Redis Cloud"""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(ENV_TEMPLATE)
        print("✅ Created .env file with Redis Cloud template")
        print("📝 Please update with your actual Redis Cloud credentials")
        return True
    else:
        print("ℹ️  .env file already exists")
        return False

def test_redis_connection():
    """Test Redis connection"""
    try:
        import redis
        config = RedisConfig.from_env()
        r = redis.Redis(**config.to_dict())
        
        # Test connection
        r.ping()
        print("✅ Redis connection successful!")
        
        # Test basic operations
        r.set("test_key", "test_value", ex=10)
        value = r.get("test_key")
        if value == "test_value":
            print("✅ Redis read/write operations working!")
        r.delete("test_key")
        
        return True
        
    except ImportError:
        print("❌ Redis library not installed. Run: pip install redis")
        return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("📝 Check your Redis Cloud credentials in .env file")
        return False

if __name__ == "__main__":
    print("🔧 Redis Cloud Financial Data Agent Configuration")
    print("=" * 50)
    
    # Create .env file if needed
    created = create_env_file()
    
    if not created:
        # Test existing configuration
        print("\n🔍 Testing Redis connection...")
        success = test_redis_connection()
        
        if success:
            config = RedisConfig.from_env()
            agent_config = AgentConfig.from_env()
            print(f"\n📊 Redis: {config.host}:{config.port}")
            print(f"🤖 Agent: {agent_config.num_batches} batches of {agent_config.batch_size}")
            print(f"🔑 Key prefix: {agent_config.key_prefix}")
            print("🚀 Ready to generate financial data!")