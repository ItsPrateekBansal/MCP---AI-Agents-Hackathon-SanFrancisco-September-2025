import redis
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict
import uuid

@dataclass
class RiskMetrics:
    customer_id: str
    credit_risk_score: float
    aml_risk_score: float
    combined_risk_score: float
    risk_factors: List[str]
    transaction_count: int
    total_amount: float
    avg_amount: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    analysis_timestamp: str
    risk_rating_id: str

@dataclass
class CustomerRating:
    """Customer Risk Rating Record for Storage"""
    rating_id: str
    customer_id: str
    risk_level: str
    combined_risk_score: float
    credit_risk_score: float
    aml_risk_score: float
    total_amount: float
    transaction_count: int
    risk_factors: List[str]
    created_at: str
    analysis_period_hours: int
    last_updated: str
    status: str  # ACTIVE, UNDER_REVIEW, ESCALATED, RESOLVED
    alert_triggered: bool
    compliance_notes: str

class UnifiedRiskScorer:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        
        # Initialize customer ratings storage
        self._initialize_rating_storage()
        
        # Risk thresholds
        self.CREDIT_RISK_THRESHOLDS = {
            'high_frequency_threshold': 10,  # transactions per hour
            'high_amount_threshold': 1000,   # single transaction
            'velocity_threshold': 5000,      # total amount per day
            'foreign_transaction_threshold': 0.3,  # 30% of transactions
            'ip_diversity_threshold': 0.5,   # 50% unique IPs
            'location_dispersion_threshold': 5,  # max different cities
            'device_diversity_threshold': 3,  # max different devices
            'session_transaction_threshold': 5  # max transactions per session
        }
        
        self.AML_RISK_THRESHOLDS = {
            'structuring_threshold': 9500,   # just under reporting limit
            'small_structuring_threshold': 3000,  # smaller structuring limit
            'round_amount_threshold': 0.7,   # 70% round amounts
            'unusual_hours_threshold': 0.4,  # 40% transactions outside business hours
            'weekend_threshold': 0.6,        # 60% weekend transactions
            'ip_diversity_threshold': 0.8,   # 80% unique IPs
            'burst_threshold': 300,          # 5 minutes for burst detection
            'rapid_transaction_threshold': 60, # 1 minute for rapid transactions
            'impossible_travel_threshold': 2,  # 2 hours minimum travel time
            'high_risk_merchant_categories': ['Cash Advance', 'Money Transfer', 'Casino', 'ATM', 'Cryptocurrency', 'Precious Metals', 'Pawn Shops'],
            'high_risk_countries': ['AF', 'BY', 'MM', 'CF', 'TD', 'CG', 'GQ', 'ER', 'HT', 'IR', 'IQ', 'LB', 'LY', 'ML', 'NI', 'KP', 'SO', 'SS', 'SD', 'SY', 'VE', 'YE', 'ZW'],
            'smurfing_small_amount_threshold': 100,  # amounts considered "small"
            'smurfing_total_threshold': 5000,       # total that triggers smurfing alert
            'ctr_avoidance_avg_threshold': 500,     # average amount for CTR avoidance
            'ctr_avoidance_total_threshold': 10000  # total amount for CTR avoidance
        }

    def _initialize_rating_storage(self):
        """Initialize Redis structures for customer rating storage"""
        # Create indexes and counters if they don't exist
        if not self.redis_client.exists("risk:ratings:counter"):
            self.redis_client.set("risk:ratings:counter", 0)
        
        # Initialize rating level counters for statistics
        for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            counter_key = f"risk:stats:level:{level}"
            if not self.redis_client.exists(counter_key):
                self.redis_client.set(counter_key, 0)

    def fetch_transactions_from_redis(self, customer_id: str = None, hours_lookback: int = 24) -> List[Dict]:
        """Fetch transaction data from Redis - main input function"""
        print(f"🔍 Fetching transactions from Redis...")
        transactions = []
        cutoff_time = datetime.now() - timedelta(hours=hours_lookback)
        
        # Pattern to match transaction keys in Redis
        pattern = "financial:transaction:TXN_*"
        keys = self.redis_client.keys(pattern)
        
        print(f"📊 Found {len(keys)} transaction keys in Redis")
        
        for key in keys:
            try:
                # Get transaction data from Redis hash
                tx_data = self.redis_client.hgetall(key)
                if not tx_data:
                    continue
                
                # Filter by customer if specified
                if customer_id and tx_data.get('customer_id') != customer_id:
                    continue
                
                # Parse and validate timestamp
                tx_time = datetime.fromisoformat(tx_data['timestamp'].replace('Z', '+00:00'))
                if tx_time < cutoff_time:
                    continue
                
                # Convert data types
                processed_tx = self._process_transaction_data(tx_data)
                if processed_tx:
                    transactions.append(processed_tx)
                    
            except (ValueError, KeyError) as e:
                print(f"⚠️  Error processing transaction {key}: {e}")
                continue
        
        print(f"✅ Successfully fetched {len(transactions)} valid transactions")
        return sorted(transactions, key=lambda x: x['timestamp'], reverse=True)

    def _process_transaction_data(self, tx_data: Dict) -> Optional[Dict]:
        """Process and validate transaction data from Redis"""
        try:
            processed = tx_data.copy()
            
            # Convert numeric fields
            processed['amount'] = float(tx_data['amount'])
            processed['risk_score'] = float(tx_data.get('risk_score', 0))
            processed['is_fraud'] = tx_data.get('is_fraud', 'false').lower() == 'true'
            
            # Convert location coordinates
            if 'location_lat' in tx_data:
                processed['location_lat'] = float(tx_data['location_lat'])
            if 'location_lon' in tx_data:
                processed['location_lon'] = float(tx_data['location_lon'])
            
            return processed
            
        except (ValueError, TypeError) as e:
            print(f"⚠️  Error converting transaction data: {e}")
            return None

    def store_customer_rating(self, risk_metrics: RiskMetrics, hours_lookback: int = 24) -> str:
        """Store customer risk rating in Redis table structure"""
        print(f"💾 Storing customer rating for {risk_metrics.customer_id}")
        
        # Generate unique rating ID
        rating_id = f"RATING_{uuid.uuid4().hex[:12].upper()}"
        current_time = datetime.now().isoformat()
        
        # Determine alert status
        alert_triggered = risk_metrics.combined_risk_score >= 60
        status = "ESCALATED" if risk_metrics.combined_risk_score >= 80 else \
                "UNDER_REVIEW" if alert_triggered else "ACTIVE"
        
        # Create customer rating record
        rating_record = CustomerRating(
            rating_id=rating_id,
            customer_id=risk_metrics.customer_id,
            risk_level=risk_metrics.risk_level,
            combined_risk_score=risk_metrics.combined_risk_score,
            credit_risk_score=risk_metrics.credit_risk_score,
            aml_risk_score=risk_metrics.aml_risk_score,
            total_amount=risk_metrics.total_amount,
            transaction_count=risk_metrics.transaction_count,
            risk_factors=risk_metrics.risk_factors,
            created_at=current_time,
            analysis_period_hours=hours_lookback,
            last_updated=current_time,
            status=status,
            alert_triggered=alert_triggered,
            compliance_notes=""
        )
        
        # Store in Redis using multiple data structures
        rating_key = f"risk:rating:{rating_id}"
        rating_data = asdict(rating_record)
        rating_data['risk_factors'] = json.dumps(rating_data['risk_factors'])
        self.redis_client.hset(rating_key, mapping=rating_data)
        
        # Customer-to-rating mapping
        customer_rating_key = f"risk:customer:{risk_metrics.customer_id}"
        self.redis_client.hset(customer_rating_key, mapping={
            "latest_rating_id": rating_id,
            "latest_risk_level": risk_metrics.risk_level,
            "latest_score": risk_metrics.combined_risk_score,
            "last_updated": current_time,
            "alert_status": "ACTIVE" if alert_triggered else "NONE"
        })
        
        # Risk level index
        risk_level_key = f"risk:index:level:{risk_metrics.risk_level}"
        self.redis_client.zadd(risk_level_key, {rating_id: risk_metrics.combined_risk_score})
        
        # Time-based index
        timestamp_score = int(datetime.now().timestamp())
        self.redis_client.zadd("risk:index:recent", {rating_id: timestamp_score})
        
        # Alert index
        if alert_triggered:
            alert_key = "risk:alerts:active"
            alert_data = {
                "rating_id": rating_id,
                "customer_id": risk_metrics.customer_id,
                "risk_level": risk_metrics.risk_level,
                "score": risk_metrics.combined_risk_score,
                "created_at": current_time
            }
            self.redis_client.hset(f"{alert_key}:{rating_id}", mapping=alert_data)
            self.redis_client.sadd("risk:alerts:list", rating_id)
        
        # Update statistics
        self._update_rating_statistics(risk_metrics.risk_level)
        
        # Set expiration (30 days)
        self.redis_client.expire(rating_key, 30 * 24 * 3600)
        
        print(f"✅ Successfully stored rating {rating_id}")
        return rating_id

    def _update_rating_statistics(self, risk_level: str):
        """Update Redis statistics counters"""
        level_counter = f"risk:stats:level:{risk_level}"
        self.redis_client.incr(level_counter)
        self.redis_client.incr("risk:ratings:counter")
        
        # Daily stats
        today = datetime.now().strftime("%Y-%m-%d")
        daily_key = f"risk:stats:daily:{today}"
        self.redis_client.hincrby(daily_key, risk_level, 1)
        self.redis_client.expire(daily_key, 7 * 24 * 3600)

    def get_customer_transactions(self, customer_id: str, hours_lookback: int = 24) -> List[Dict]:
        """Retrieve recent transactions for a customer from Redis"""
        return self.fetch_transactions_from_redis(customer_id, hours_lookback)

    def calculate_unified_risk(self, customer_id: str, hours_lookback: int = 24) -> RiskMetrics:
        """Calculate unified credit + AML risk score for a customer and store rating"""
        print(f"🎯 Calculating unified risk for customer {customer_id}")
        
        # Fetch transactions from Redis
        transactions = self.fetch_transactions_from_redis(customer_id, hours_lookback)
        
        if not transactions:
            return RiskMetrics(
                customer_id=customer_id,
                credit_risk_score=0.0,
                aml_risk_score=0.0,
                combined_risk_score=0.0,
                risk_factors=[],
                transaction_count=0,
                total_amount=0.0,
                avg_amount=0.0,
                risk_level="LOW",
                analysis_timestamp=datetime.now().isoformat(),
                risk_rating_id=""
            )
        
        # Calculate individual risk scores
        credit_score, credit_factors = self.calculate_credit_risk_score(transactions)
        aml_score, aml_factors = self.calculate_aml_risk_score(transactions)
        
        # Combined risk score (weighted average)
        combined_score = (credit_score * 0.4) + (aml_score * 0.6)
        
        # Calculate transaction statistics
        amounts = [tx['amount'] for tx in transactions]
        total_amount = sum(amounts)
        avg_amount = total_amount / len(amounts)
        
        # Determine risk level
        if combined_score >= 80:
            risk_level = "CRITICAL"
        elif combined_score >= 60:
            risk_level = "HIGH"
        elif combined_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Create risk metrics
        risk_metrics = RiskMetrics(
            customer_id=customer_id,
            credit_risk_score=credit_score,
            aml_risk_score=aml_score,
            combined_risk_score=combined_score,
            risk_factors=credit_factors + aml_factors,
            transaction_count=len(transactions),
            total_amount=total_amount,
            avg_amount=avg_amount,
            risk_level=risk_level,
            analysis_timestamp=datetime.now().isoformat(),
            risk_rating_id=""
        )
        
        # Store customer rating in Redis
        rating_id = self.store_customer_rating(risk_metrics, hours_lookback)
        risk_metrics.risk_rating_id = rating_id
        
        print(f"✅ Risk analysis complete - Score: {combined_score:.1f}")
        return risk_metrics

    def calculate_credit_risk_score(self, transactions: List[Dict]) -> Tuple[float, List[str]]:
        """Calculate credit risk score based on comprehensive patterns"""
        if not transactions:
            return 0.0, []
        
        risk_factors = []
        risk_score = 0.0
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Transaction Frequency Risk
        hourly_tx_count = len(df) / 24
        if hourly_tx_count > self.CREDIT_RISK_THRESHOLDS['high_frequency_threshold']:
            risk_score += 25
            risk_factors.append(f"High transaction frequency: {hourly_tx_count:.1f} tx/hour")
        
        # High Amount Transactions
        high_amount_txs = df[df['amount'] > self.CREDIT_RISK_THRESHOLDS['high_amount_threshold']]
        if len(high_amount_txs) > 0:
            risk_score += 20
            risk_factors.append(f"High-value transactions: {len(high_amount_txs)} > ${self.CREDIT_RISK_THRESHOLDS['high_amount_threshold']}")
        
        # Spending Velocity
        total_amount = df['amount'].sum()
        if total_amount > self.CREDIT_RISK_THRESHOLDS['velocity_threshold']:
            risk_score += 30
            risk_factors.append(f"High spending velocity: ${total_amount:.2f} in 24h")
        
        # Geographic Analysis
        foreign_txs = df[df['location_country'] != 'US']
        foreign_ratio = len(foreign_txs) / len(df)
        if foreign_ratio > self.CREDIT_RISK_THRESHOLDS['foreign_transaction_threshold']:
            risk_score += 15
            risk_factors.append(f"High foreign transaction ratio: {foreign_ratio:.1%}")
        
        # IP Address Analysis
        unique_ips = df['ip_address'].nunique()
        ip_ratio = unique_ips / len(df)
        if ip_ratio > self.CREDIT_RISK_THRESHOLDS['ip_diversity_threshold']:
            risk_score += 25
            risk_factors.append(f"High IP diversity: {unique_ips} different IPs")
        
        return min(risk_score, 100.0), risk_factors

    def calculate_aml_risk_score(self, transactions: List[Dict]) -> Tuple[float, List[str]]:
        """Calculate AML risk score based on suspicious patterns"""
        if not transactions:
            return 0.0, []
        
        risk_factors = []
        risk_score = 0.0
        
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Structuring Detection
        structuring_txs = df[(df['amount'] > 9000) & (df['amount'] < self.AML_RISK_THRESHOLDS['structuring_threshold'])]
        if len(structuring_txs) > 2:
            risk_score += 40
            risk_factors.append(f"Potential structuring: {len(structuring_txs)} transactions near $10K")
        
        # Round Amount Patterns
        round_amounts = df[df['amount'] % 100 == 0]
        round_ratio = len(round_amounts) / len(df)
        if round_ratio > self.AML_RISK_THRESHOLDS['round_amount_threshold']:
            risk_score += 25
            risk_factors.append(f"Suspicious round amounts: {round_ratio:.1%}")
        
        # Unusual Time Patterns
        unusual_hours = df[(df['hour'] < 6) | (df['hour'] > 22)]
        unusual_ratio = len(unusual_hours) / len(df)
        if unusual_ratio > self.AML_RISK_THRESHOLDS['unusual_hours_threshold']:
            risk_score += 20
            risk_factors.append(f"Unusual timing: {unusual_ratio:.1%} outside business hours")
        
        # High-Risk Merchants
        risky_merchant_txs = df[df['merchant_category'].isin(self.AML_RISK_THRESHOLDS['high_risk_merchant_categories'])]
        if len(risky_merchant_txs) > 0:
            risk_score += 30
            risk_factors.append(f"High-risk merchants: {len(risky_merchant_txs)} risky transactions")
        
        return min(risk_score, 100.0), risk_factors

    def get_top_risky_customers(self, limit: int = 20, hours_lookback: int = 24) -> List[RiskMetrics]:
        """Get top risky customers from stored ratings"""
        print(f"🔍 Getting top {limit} risky customers...")
        
        # Get all transactions and process customers
        all_transactions = self.fetch_transactions_from_redis(None, hours_lookback)
        customer_ids = set(tx['customer_id'] for tx in all_transactions)
        
        risk_metrics = []
        for customer_id in customer_ids:
            try:
                metrics = self.calculate_unified_risk(customer_id, hours_lookback)
                if metrics.transaction_count > 0:
                    risk_metrics.append(metrics)
            except Exception as e:
                print(f"❌ Error calculating risk for {customer_id}: {e}")
                continue
        
        # Sort by risk score
        risk_metrics.sort(key=lambda x: x.combined_risk_score, reverse=True)
        return risk_metrics[:limit]

    def process_all_customers(self, hours_lookback: int = 24) -> Dict:
        """Process all customers and store their risk ratings"""
        print(f"🚀 Starting batch risk analysis...")
        
        all_transactions = self.fetch_transactions_from_redis(None, hours_lookback)
        customer_ids = set(tx['customer_id'] for tx in all_transactions)
        
        results = {
            "processed": 0,
            "alerts_generated": 0,
            "risk_levels": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "errors": 0
        }
        
        for customer_id in customer_ids:
            try:
                metrics = self.calculate_unified_risk(customer_id, hours_lookback)
                results["processed"] += 1
                results["risk_levels"][metrics.risk_level] += 1
                
                if metrics.combined_risk_score >= 60:
                    results["alerts_generated"] += 1
                    
            except Exception as e:
                print(f"❌ Error processing {customer_id}: {e}")
                results["errors"] += 1
        
        print(f"✅ Processed {results['processed']} customers")
        return results

# Example usage
if __name__ == "__main__":
    scorer = UnifiedRiskScorer()
    
    print("🚀 Processing all customers...")
    results = scorer.process_all_customers()
    
    print("\n📊 Getting top risky customers...")
    top_risky = scorer.get_top_risky_customers(limit=5)
    
    for i, metrics in enumerate(top_risky, 1):
        print(f"\n{i}. Customer: {metrics.customer_id}")
        print(f"   Score: {metrics.combined_risk_score:.1f} ({metrics.risk_level})")
        print(f"   Rating ID: {metrics.risk_rating_id}")