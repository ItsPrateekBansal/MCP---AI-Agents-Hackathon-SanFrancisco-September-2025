import redis
import json
import uuid
import orjson
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
import logging
from faker import Faker
import random
import ipaddress
from tqdm import tqdm
import time

# Import our configuration
from redis_config import RedisConfig, AgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TransactionData:
    transaction_id: str
    customer_id: str
    amount: float
    currency: str
    transaction_type: str
    merchant_name: str
    merchant_category: str
    timestamp: str  # ISO format string for Redis
    ip_address: str
    location_country: str
    location_city: str
    location_lat: float
    location_lon: float
    device_type: str
    user_agent: str
    payment_method: str
    card_last_four: str
    risk_score: float
    is_fraud: bool
    session_id: str

class FinancialDataGenerator:
    def __init__(self, config: AgentConfig):
        self.fake = Faker()
        self.config = config
        
        # Risk patterns for more realistic data
        self.high_risk_countries = ['NG', 'PK', 'BD', 'ID', 'IN']
        self.high_risk_merchants = ['Online Gaming', 'Cryptocurrency', 'Adult Entertainment']
        self.suspicious_amounts = [999.99, 1000.00, 2500.00, 4999.99]
        
        # Common merchant categories
        self.merchant_categories = [
            'Grocery', 'Gas Station', 'Restaurant', 'Online Retail', 'Department Store',
            'ATM Withdrawal', 'Bank Transfer', 'Subscription Service', 'Travel',
            'Healthcare', 'Entertainment', 'Utilities', 'Insurance'
        ]
        
        # Device types and user agents
        self.device_types = ['Mobile', 'Desktop', 'Tablet']
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Mozilla/5.0 (Android 11; Mobile; rv:68.0)',
        ]

    def generate_ip_address(self, risk_level: str = 'normal') -> str:
        """Generate IP address with geographic patterns"""
        if risk_level == 'high':
            ranges = ['41.0.0.0/8', '103.0.0.0/8', '185.0.0.0/8']
            network = ipaddress.IPv4Network(random.choice(ranges))
            return str(network[random.randint(1, min(1000, network.num_addresses - 2))])
        else:
            ranges = ['192.168.0.0/16', '10.0.0.0/8', '172.16.0.0/12']
            network = ipaddress.IPv4Network(random.choice(ranges))
            return str(network[random.randint(1, min(1000, network.num_addresses - 2))])

    def calculate_risk_score(self, transaction: Dict) -> float:
        """Calculate risk score based on transaction attributes"""
        score = 0.1  # Base score
        
        # Amount-based risk
        if transaction['amount'] > 5000:
            score += 0.3
        elif transaction['amount'] in self.suspicious_amounts:
            score += 0.4
        elif transaction['amount'] < 1:
            score += 0.2
            
        # Location-based risk
        if transaction['location_country'] in self.high_risk_countries:
            score += 0.3
            
        # Merchant category risk
        if transaction['merchant_category'] in self.high_risk_merchants:
            score += 0.2
            
        # Time-based risk (late night transactions)
        timestamp = datetime.fromisoformat(transaction['timestamp'])
        hour = timestamp.hour
        if hour < 6 or hour > 23:
            score += 0.1
            
        return min(score, 1.0)

    def generate_transaction(self, customer_id: Optional[str] = None) -> TransactionData:
        """Generate a single realistic transaction"""
        
        if not customer_id:
            customer_id = f"CUST_{uuid.uuid4().hex[:8].upper()}"
        
        transaction_type = random.choices(
            ['purchase', 'withdrawal', 'transfer', 'payment'],
            weights=[70, 15, 10, 5]
        )[0]
        
        amount = self._generate_realistic_amount(transaction_type)
        currency = random.choices(['USD', 'EUR', 'GBP', 'CAD'], weights=[60, 20, 15, 5])[0]
        
        merchant_category = random.choice(self.merchant_categories)
        merchant_name = self.fake.company()
        
        location_data = self._generate_location()
        
        device_type = random.choice(self.device_types)
        risk_level = 'high' if random.random() < self.config.high_risk_rate else 'normal'
        ip_address = self.generate_ip_address(risk_level)
        
        timestamp = self.fake.date_time_between(start_date='-30d', end_date='now')
        
        transaction_dict = {
            'transaction_id': f"TXN_{uuid.uuid4().hex[:12].upper()}",
            'customer_id': customer_id,
            'amount': amount,
            'currency': currency,
            'transaction_type': transaction_type,
            'merchant_name': merchant_name,
            'merchant_category': merchant_category,
            'timestamp': timestamp.isoformat(),
            'ip_address': ip_address,
            'location_country': location_data['country'],
            'location_city': location_data['city'],
            'location_lat': location_data['lat'],
            'location_lon': location_data['lon'],
            'device_type': device_type,
            'user_agent': random.choice(self.user_agents),
            'payment_method': random.choices(
                ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet'],
                weights=[40, 30, 20, 10]
            )[0],
            'card_last_four': str(random.randint(1000, 9999)),
            'session_id': f"SESS_{uuid.uuid4().hex[:16]}",
        }
        
        risk_score = self.calculate_risk_score(transaction_dict)
        is_fraud = risk_score > 0.7 and random.random() < (self.config.fraud_rate * 6)
        
        return TransactionData(
            **transaction_dict,
            risk_score=round(risk_score, 3),
            is_fraud=is_fraud
        )

    def _generate_realistic_amount(self, transaction_type: str) -> float:
        """Generate realistic amounts based on transaction type"""
        if transaction_type == 'purchase':
            return round(random.lognormvariate(3.5, 1.2), 2)
        elif transaction_type == 'withdrawal':
            return round(random.choice([20, 40, 60, 80, 100, 200]), 2)
        elif transaction_type == 'transfer':
            return round(random.lognormvariate(5.0, 1.5), 2)
        else:
            return round(random.lognormvariate(4.0, 1.0), 2)

    def _generate_location(self) -> Dict:
        """Generate realistic location data"""
        locations = [
            {'country': 'US', 'city': 'New York', 'lat': 40.7128, 'lon': -74.0060},
            {'country': 'US', 'city': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
            {'country': 'US', 'city': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
            {'country': 'GB', 'city': 'London', 'lat': 51.5074, 'lon': -0.1278},
            {'country': 'CA', 'city': 'Toronto', 'lat': 43.6532, 'lon': -79.3832},
            {'country': 'DE', 'city': 'Berlin', 'lat': 52.5200, 'lon': 13.4050},
            {'country': 'AU', 'city': 'Sydney', 'lat': -33.8688, 'lon': 151.2093},
        ]
        
        if random.random() < self.config.high_risk_rate:
            high_risk_locations = [
                {'country': 'NG', 'city': 'Lagos', 'lat': 6.5244, 'lon': 3.3792},
                {'country': 'PK', 'city': 'Karachi', 'lat': 24.8607, 'lon': 67.0011},
                {'country': 'BD', 'city': 'Dhaka', 'lat': 23.8103, 'lon': 90.4125},
            ]
            return random.choice(high_risk_locations)
        
        return random.choice(locations)

    def generate_batch(self, count: int, customer_ids: List[str] = None) -> List[TransactionData]:
        """Generate a batch of transactions"""
        transactions = []
        
        for i in range(count):
            customer_id = None
            if customer_ids:
                customer_id = random.choice(customer_ids)
            
            transaction = self.generate_transaction(customer_id)
            transactions.append(transaction)
            
        return transactions

class RedisDataManager:
    def __init__(self, redis_config: RedisConfig, agent_config: AgentConfig):
        self.redis_config = redis_config
        self.agent_config = agent_config
        self.redis_client = None
        
        # Redis key patterns
        self.key_prefix = agent_config.key_prefix
        self.transaction_key_pattern = f"{self.key_prefix}:transaction:"
        self.customer_key_pattern = f"{self.key_prefix}:customer:"
        self.stats_key_pattern = f"{self.key_prefix}:stats:"
        self.risk_key_pattern = f"{self.key_prefix}:risk:"
        
    def connect(self):
        """Establish Redis connection"""
        try:
            self.redis_client = redis.Redis(**self.redis_config.to_dict())
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def store_transactions(self, transactions: List[TransactionData]) -> Dict[str, int]:
        """Store transactions in Redis with multiple data structures"""
        stats = {
            'stored': 0,
            'errors': 0,
            'high_risk': 0,
            'fraud': 0
        }
        
        if self.agent_config.use_pipeline:
            pipe = self.redis_client.pipeline()
            
            for i, transaction in enumerate(transactions):
                try:
                    self._store_single_transaction(transaction, pipe)
                    stats['stored'] += 1
                    
                    if transaction.is_fraud:
                        stats['fraud'] += 1
                    elif transaction.risk_score > 0.5:
                        stats['high_risk'] += 1
                    
                    # Execute pipeline in batches
                    if (i + 1) % self.agent_config.pipeline_size == 0:
                        pipe.execute()
                        pipe = self.redis_client.pipeline()
                        
                except Exception as e:
                    logger.error(f"Error storing transaction {transaction.transaction_id}: {e}")
                    stats['errors'] += 1
            
            # Execute remaining transactions
            if len(transactions) % self.agent_config.pipeline_size != 0:
                pipe.execute()
        else:
            # Store individually
            for transaction in transactions:
                try:
                    self._store_single_transaction(transaction)
                    stats['stored'] += 1
                    
                    if transaction.is_fraud:
                        stats['fraud'] += 1
                    elif transaction.risk_score > 0.5:
                        stats['high_risk'] += 1
                        
                except Exception as e:
                    logger.error(f"Error storing transaction {transaction.transaction_id}: {e}")
                    stats['errors'] += 1
        
        return stats
    
    def _store_single_transaction(self, transaction: TransactionData, pipe=None):
        """Store a single transaction with multiple Redis data structures"""
        client = pipe if pipe else self.redis_client
        
        # Convert transaction to dict and serialize
        tx_dict = asdict(transaction)
        tx_json = orjson.dumps(tx_dict).decode('utf-8')
        
        # 1. Store main transaction data (Hash)
        tx_key = f"{self.transaction_key_pattern}{transaction.transaction_id}"
        client.hset(tx_key, mapping={
            'data': tx_json,
            'customer_id': transaction.customer_id,
            'amount': transaction.amount,
            'risk_score': transaction.risk_score,
            'is_fraud': str(transaction.is_fraud),
            'timestamp': transaction.timestamp,
            'location_country': transaction.location_country
        })
        client.expire(tx_key, self.agent_config.transaction_ttl)
        
        # 2. Add to customer's transaction list (List)
        customer_key = f"{self.customer_key_pattern}{transaction.customer_id}"
        client.lpush(customer_key, transaction.transaction_id)
        client.expire(customer_key, self.agent_config.transaction_ttl)
        
        # 3. Add to risk-based sorted sets
        if transaction.is_fraud:
            client.zadd(f"{self.risk_key_pattern}fraud", {transaction.transaction_id: time.time()})
        
        if transaction.risk_score > 0.5:
            client.zadd(f"{self.risk_key_pattern}high_risk", 
                       {transaction.transaction_id: transaction.risk_score})
        
        # 4. Update daily stats (Hash)
        date_key = datetime.fromisoformat(transaction.timestamp).strftime('%Y-%m-%d')
        stats_key = f"{self.stats_key_pattern}daily:{date_key}"
        
        client.hincrby(stats_key, 'transaction_count', 1)
        client.hincrbyfloat(stats_key, 'total_amount', transaction.amount)
        client.sadd(f"{stats_key}:customers", transaction.customer_id)
        
        if transaction.is_fraud:
            client.hincrby(stats_key, 'fraud_count', 1)
        
        client.expire(stats_key, self.agent_config.transaction_ttl)
        
        # 5. Country-based tracking (Set)
        country_key = f"{self.key_prefix}:country:{transaction.location_country}"
        client.sadd(country_key, transaction.transaction_id)
        client.expire(country_key, self.agent_config.transaction_ttl)
    
    def get_statistics(self) -> Dict:
        """Get overall statistics from Redis"""
        try:
            stats = {}
            
            # Count different types of data
            stats['total_transactions'] = len(self.redis_client.keys(f"{self.transaction_key_pattern}*"))
            stats['total_customers'] = len(self.redis_client.keys(f"{self.customer_key_pattern}*"))
            stats['fraud_transactions'] = self.redis_client.zcard(f"{self.risk_key_pattern}fraud")
            stats['high_risk_transactions'] = self.redis_client.zcard(f"{self.risk_key_pattern}high_risk")
            
            # Recent high-risk transactions
            recent_high_risk = self.redis_client.zrevrange(
                f"{self.risk_key_pattern}high_risk", 0, 9, withscores=True
            )
            stats['recent_high_risk'] = [
                {'transaction_id': tx_id, 'risk_score': score} 
                for tx_id, score in recent_high_risk
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")

class RedisFinancialDataAgent:
    def __init__(self, redis_config: RedisConfig, agent_config: AgentConfig):
        self.redis_config = redis_config
        self.agent_config = agent_config
        self.generator = FinancialDataGenerator(agent_config)
        self.data_manager = RedisDataManager(redis_config, agent_config)
        
    def run(self, customer_ids: List[str] = None):
        """Main agent execution method"""
        try:
            # Connect to Redis
            self.data_manager.connect()
            
            # Generate customer IDs if not provided
            if not customer_ids:
                customer_ids = [f"CUST_{uuid.uuid4().hex[:8].upper()}" 
                              for _ in range(self.agent_config.num_customers)]
            
            total_stats = {
                'stored': 0,
                'errors': 0,
                'high_risk': 0,
                'fraud': 0
            }
            
            logger.info(f"🚀 Starting Redis Financial Data Agent")
            logger.info(f"📊 Generating {self.agent_config.num_batches} batches of {self.agent_config.batch_size}")
            
            # Generate and store data in batches
            for batch_num in tqdm(range(self.agent_config.num_batches), desc="Processing batches"):
                # Generate transaction batch
                transactions = self.generator.generate_batch(
                    self.agent_config.batch_size, 
                    customer_ids
                )
                
                # Store in Redis
                batch_stats = self.data_manager.store_transactions(transactions)
                
                # Update totals
                for key in total_stats:
                    total_stats[key] += batch_stats[key]
                
                logger.info(f"Batch {batch_num + 1}: {batch_stats}")
            
            # Get final statistics
            final_stats = self.data_manager.get_statistics()
            
            logger.info("🎉 Agent completed successfully!")
            logger.info(f"📈 Total stored: {total_stats['stored']}")
            logger.info(f"🚨 Fraud transactions: {total_stats['fraud']}")
            logger.info(f"⚠️  High risk transactions: {total_stats['high_risk']}")
            logger.info(f"❌ Errors: {total_stats['errors']}")
            
            return total_stats, final_stats
            
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}")
            raise
        finally:
            self.data_manager.close()

# Example usage
if __name__ == "__main__":
    from redis_config import RedisConfig, AgentConfig
    
    # Load configurations
    redis_config = RedisConfig.from_env()
    agent_config = AgentConfig.from_env()
    
    # Create and run the agent
    agent = RedisFinancialDataAgent(redis_config, agent_config)
    
    # Run the agent
    total_stats, final_stats = agent.run()
    
    print("\n" + "="*50)
    print("📊 FINAL STATISTICS")
    print("="*50)
    for key, value in final_stats.items():
        print(f"{key}: {value}")