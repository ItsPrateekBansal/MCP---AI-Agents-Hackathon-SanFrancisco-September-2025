# risk_scoring_agent.py - Advanced Risk Scoring System
import json
import redis
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib
import statistics
from redis_config import RedisConfig, AgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CustomerRiskProfile:
    """Customer risk profile data structure"""
    customer_id: str
    credit_score: float = 0.0
    aml_score: float = 0.0
    combined_score: float = 0.0
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    last_updated: str = ""
    total_transactions: int = 0
    total_amount: float = 0.0
    avg_transaction_amount: float = 0.0
    unique_merchants: int = 0
    unique_locations: int = 0
    unique_ips: int = 0
    velocity_flags: int = 0
    location_anomalies: int = 0
    amount_anomalies: int = 0
    merchant_risk_flags: int = 0
    behavioral_flags: int = 0
    fraud_indicators: List[str] = None
    
    def __post_init__(self):
        if self.fraud_indicators is None:
            self.fraud_indicators = []

class RiskScoringAgent:
    """Advanced Risk Scoring Agent for Credit and AML Analysis"""
    
    def __init__(self, redis_config: RedisConfig, agent_config: AgentConfig):
        self.redis_client = redis.Redis(**redis_config.to_dict())
        self.config = agent_config
        self.key_prefix = agent_config.key_prefix
        
        # Risk scoring weights
        self.weights = {
            'velocity': 0.25,
            'location': 0.20,
            'amount': 0.15,
            'merchant': 0.15,
            'behavioral': 0.15,
            'ip': 0.10
        }
        
        # Risk thresholds
        self.thresholds = {
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 0.8,
            'CRITICAL': 1.0
        }
        
    def get_customer_transactions(self, customer_id: str, days: int = 30) -> List[Dict]:
        """Fetch customer transactions from Redis"""
        try:
            # Get all transaction keys - your pattern is financial:transaction:TXN_*
            pattern = f"{self.key_prefix}:transaction:*"
            transaction_keys = self.redis_client.keys(pattern)
            
            customer_transactions = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            logger.info(f"Found {len(transaction_keys)} total transaction keys, filtering for customer {customer_id}")
            
            for key in transaction_keys:
                try:
                    # Handle key as bytes or string
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                    
                    # First try as hash
                    transaction_data = self.redis_client.hgetall(key)
                    
                    if not transaction_data:
                        # Try as string (JSON)
                        raw_data = self.redis_client.get(key)
                        if raw_data:
                            raw_str = raw_data.decode('utf-8') if isinstance(raw_data, bytes) else str(raw_data)
                            try:
                                transaction = json.loads(raw_str)
                            except json.JSONDecodeError:
                                logger.debug(f"Key {key_str} is not valid JSON")
                                continue
                        else:
                            continue
                    else:
                        # Parse hash data
                        transaction = {}
                        for k, v in transaction_data.items():
                            k_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                            v_str = v.decode('utf-8') if isinstance(v, bytes) else str(v)
                            transaction[k_str] = v_str
                    
                    # Check if transaction belongs to customer
                    if transaction.get('customer_id') != customer_id:
                        continue
                    
                    # Check timestamp and filter by date
                    timestamp_str = transaction.get('timestamp', '')
                    if not timestamp_str:
                        continue
                        
                    # Handle different timestamp formats
                    try:
                        # Try parsing with different formats
                        if 'T' in timestamp_str:
                            if timestamp_str.endswith('Z'):
                                tx_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            else:
                                tx_time = datetime.fromisoformat(timestamp_str)
                        else:
                            tx_time = datetime.fromisoformat(timestamp_str)
                    except ValueError as e:
                        logger.debug(f"Could not parse timestamp '{timestamp_str}': {e}")
                        continue
                    
                    if tx_time < cutoff_date:
                        continue
                    
                    # Convert numeric fields safely
                    try:
                        transaction['amount'] = float(transaction.get('amount', 0))
                        transaction['risk_score'] = float(transaction.get('risk_score', 0))
                        transaction['location_lat'] = float(transaction.get('location_lat', 0))
                        transaction['location_lon'] = float(transaction.get('location_lon', 0))
                        transaction['is_fraud'] = str(transaction.get('is_fraud', 'false')).lower() == 'true'
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error converting numeric fields for {key_str}: {e}")
                        # Set defaults for failed conversions
                        transaction['amount'] = float(transaction.get('amount', 0)) if transaction.get('amount', '').replace('.','').isdigit() else 0.0
                        transaction['risk_score'] = 0.0
                        transaction['location_lat'] = 0.0
                        transaction['location_lon'] = 0.0
                        transaction['is_fraud'] = False
                    
                    customer_transactions.append(transaction)
                            
                except Exception as e:
                    logger.debug(f"Error parsing transaction {key}: {e}")
                    continue
                    
            logger.info(f"Found {len(customer_transactions)} transactions for customer {customer_id}")
            return customer_transactions
            
        except Exception as e:
            logger.error(f"Error fetching transactions for {customer_id}: {e}")
            return []
    
    def calculate_velocity_score(self, transactions: List[Dict]) -> Tuple[float, List[str]]:
        """Calculate velocity-based risk score"""
        if not transactions:
            return 0.0, []
            
        flags = []
        score = 0.0
        
        # Sort by timestamp
        transactions.sort(key=lambda x: x['timestamp'])
        
        # Check for rapid-fire transactions
        rapid_transactions = 0
        for i in range(1, len(transactions)):
            prev_time = datetime.fromisoformat(transactions[i-1]['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
            curr_time = datetime.fromisoformat(transactions[i]['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
            
            time_diff = (curr_time - prev_time).total_seconds()
            if time_diff < 60:  # Less than 1 minute apart
                rapid_transactions += 1
                
        if rapid_transactions > 5:
            score += 0.4
            flags.append(f"Rapid transactions detected: {rapid_transactions} within 1 minute")
            
        # Check daily transaction volume
        daily_counts = defaultdict(int)
        daily_amounts = defaultdict(float)
        
        for tx in transactions:
            date = tx['timestamp'][:10]
            daily_counts[date] += 1
            daily_amounts[date] += tx['amount']
            
        max_daily_count = max(daily_counts.values()) if daily_counts else 0
        max_daily_amount = max(daily_amounts.values()) if daily_amounts else 0
        
        if max_daily_count > 20:
            score += 0.3
            flags.append(f"High daily transaction count: {max_daily_count}")
            
        if max_daily_amount > 10000:
            score += 0.3
            flags.append(f"High daily amount: ${max_daily_amount:,.2f}")
            
        return min(score, 1.0), flags
    
    def calculate_location_score(self, transactions: List[Dict]) -> Tuple[float, List[str]]:
        """Calculate location-based risk score"""
        if not transactions:
            return 0.0, []
            
        flags = []
        score = 0.0
        
        locations = [(tx['location_lat'], tx['location_lon']) for tx in transactions]
        unique_locations = len(set(locations))
        
        # Check for impossible travel
        impossible_travel = 0
        sorted_txs = sorted(transactions, key=lambda x: x['timestamp'])
        
        for i in range(1, len(sorted_txs)):
            prev_tx = sorted_txs[i-1]
            curr_tx = sorted_txs[i]
            
            # Calculate distance (simple haversine approximation)
            lat1, lon1 = prev_tx['location_lat'], prev_tx['location_lon']
            lat2, lon2 = curr_tx['location_lat'], curr_tx['location_lon']
            
            distance = self._calculate_distance(lat1, lon1, lat2, lon2)
            
            # Time difference in hours
            prev_time = datetime.fromisoformat(prev_tx['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
            curr_time = datetime.fromisoformat(curr_tx['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
            time_diff_hours = (curr_time - prev_time).total_seconds() / 3600
            
            # Check if travel speed is impossible (> 500 mph)
            if time_diff_hours > 0 and distance / time_diff_hours > 500:
                impossible_travel += 1
                
        if impossible_travel > 0:
            score += 0.6
            flags.append(f"Impossible travel detected: {impossible_travel} instances")
            
        # Check for too many unique locations
        if unique_locations > 10:
            score += 0.3
            flags.append(f"High location diversity: {unique_locations} unique locations")
            
        # Check for high-risk countries/regions
        high_risk_countries = ['XX', 'YY']  # Add actual high-risk country codes
        risky_locations = sum(1 for tx in transactions if tx.get('location_country') in high_risk_countries)
        
        if risky_locations > 0:
            score += 0.4
            flags.append(f"Transactions from high-risk locations: {risky_locations}")
            
        return min(score, 1.0), flags
    
    def calculate_amount_score(self, transactions: List[Dict]) -> Tuple[float, List[str]]:
        """Calculate amount-based risk score"""
        if not transactions:
            return 0.0, []
            
        flags = []
        score = 0.0
        
        amounts = [tx['amount'] for tx in transactions]
        if not amounts:
            return 0.0, []
            
        avg_amount = statistics.mean(amounts)
        median_amount = statistics.median(amounts)
        std_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0
        max_amount = max(amounts)
        
        # Check for structuring (amounts just under reporting thresholds)
        structuring_count = sum(1 for amt in amounts if 9000 <= amt <= 9999)
        if structuring_count > 2:
            score += 0.5
            flags.append(f"Potential structuring: {structuring_count} transactions near $10K threshold")
            
        # Check for unusually large transactions
        if max_amount > avg_amount + 3 * std_amount and max_amount > 5000:
            score += 0.3
            flags.append(f"Unusually large transaction: ${max_amount:,.2f} vs avg ${avg_amount:,.2f}")
            
        # Check for round number bias (money laundering indicator)
        round_numbers = sum(1 for amt in amounts if amt % 100 == 0 and amt >= 1000)
        if round_numbers > len(amounts) * 0.3:
            score += 0.2
            flags.append(f"High round number frequency: {round_numbers}/{len(amounts)} transactions")
            
        # Check for credit risk patterns
        if avg_amount > 1000 and len([amt for amt in amounts if amt > avg_amount * 2]) > 3:
            score += 0.3
            flags.append("Credit risk: Pattern of increasing transaction amounts")
            
        return min(score, 1.0), flags
    
    def calculate_merchant_score(self, transactions: List[Dict]) -> Tuple[float, List[str]]:
        """Calculate merchant-based risk score"""
        if not transactions:
            return 0.0, []
            
        flags = []
        score = 0.0
        
        merchant_counts = Counter(tx.get('merchant_name', 'Unknown') for tx in transactions)
        merchant_categories = Counter(tx.get('merchant_category', 'Unknown') for tx in transactions)
        
        # Check for high-risk merchant categories
        high_risk_categories = [
            'Money Transfer', 'Cryptocurrency', 'Adult Entertainment',
            'Gambling', 'Check Cashing', 'Pawn Shops'
        ]
        
        risky_category_txs = sum(count for category, count in merchant_categories.items() 
                               if any(risk_cat in category for risk_cat in high_risk_categories))
        
        if risky_category_txs > 0:
            score += 0.4
            flags.append(f"High-risk merchant categories: {risky_category_txs} transactions")
            
        # Check for merchant concentration (potential money laundering)
        if merchant_counts:
            max_merchant_count = max(merchant_counts.values())
            total_transactions = len(transactions)
            
            if max_merchant_count > total_transactions * 0.7:
                score += 0.3
                most_used_merchant = max(merchant_counts, key=merchant_counts.get)
                flags.append(f"High merchant concentration: {max_merchant_count}/{total_transactions} with {most_used_merchant}")
                
        # Check for unusual merchant diversity (credit risk)
        unique_merchants = len(merchant_counts)
        if unique_merchants > 15 and len(transactions) < 30:
            score += 0.2
            flags.append(f"High merchant diversity: {unique_merchants} merchants in {len(transactions)} transactions")
            
        return min(score, 1.0), flags
    
    def calculate_behavioral_score(self, transactions: List[Dict]) -> Tuple[float, List[str]]:
        """Calculate behavioral pattern risk score"""
        if not transactions:
            return 0.0, []
            
        flags = []
        score = 0.0
        
        # Analyze time patterns
        hours = [datetime.fromisoformat(tx['timestamp'].replace('Z', '+00:00').replace('+00:00', '')).hour 
                for tx in transactions]
        
        # Check for unusual timing patterns
        night_transactions = sum(1 for hour in hours if hour < 6 or hour > 23)
        if night_transactions > len(transactions) * 0.3:
            score += 0.2
            flags.append(f"High night-time activity: {night_transactions}/{len(transactions)} transactions")
            
        # Check device/user agent patterns
        devices = Counter(tx.get('device_type', 'Unknown') for tx in transactions)
        user_agents = Counter(tx.get('user_agent', 'Unknown') for tx in transactions)
        
        # Multiple device types might indicate account sharing or compromise
        if len(devices) > 3:
            score += 0.2
            flags.append(f"Multiple device types: {len(devices)} different devices")
            
        # Check payment method diversity
        payment_methods = Counter(tx.get('payment_method', 'Unknown') for tx in transactions)
        if len(payment_methods) > 3:
            score += 0.1
            flags.append(f"Multiple payment methods: {len(payment_methods)} different methods")
            
        # Check for session pattern anomalies
        sessions = Counter(tx.get('session_id', 'Unknown') for tx in transactions)
        avg_txs_per_session = len(transactions) / len(sessions) if sessions else 0
        
        if avg_txs_per_session > 10:
            score += 0.2
            flags.append(f"High transactions per session: {avg_txs_per_session:.1f} avg")
            
        return min(score, 1.0), flags
    
    def calculate_ip_score(self, transactions: List[Dict]) -> Tuple[float, List[str]]:
        """Calculate IP-based risk score"""
        if not transactions:
            return 0.0, []
            
        flags = []
        score = 0.0
        
        ips = [tx.get('ip_address', 'Unknown') for tx in transactions]
        unique_ips = len(set(ips))
        
        # Check for too many unique IPs
        if unique_ips > 10:
            score += 0.3
            flags.append(f"High IP diversity: {unique_ips} unique IP addresses")
            
        # Check for suspicious IP patterns (simplified - in production, use IP geolocation services)
        private_ips = sum(1 for ip in ips if ip.startswith(('192.168.', '10.', '172.')))
        if private_ips > len(transactions) * 0.8:
            score += 0.2
            flags.append(f"High private IP usage: {private_ips}/{len(transactions)}")
            
        # Check for rapid IP switching
        ip_switches = 0
        sorted_txs = sorted(transactions, key=lambda x: x['timestamp'])
        for i in range(1, len(sorted_txs)):
            if sorted_txs[i].get('ip_address') != sorted_txs[i-1].get('ip_address'):
                ip_switches += 1
                
        if ip_switches > len(transactions) * 0.5:
            score += 0.3
            flags.append(f"Frequent IP switching: {ip_switches} switches")
            
        return min(score, 1.0), flags
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using haversine formula"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 3956  # Radius of earth in miles
        
        return c * r
    
    def calculate_customer_risk_score(self, customer_id: str) -> CustomerRiskProfile:
        """Calculate comprehensive risk score for a customer"""
        logger.info(f"Calculating risk score for customer: {customer_id}")
        
        # Get customer transactions
        transactions = self.get_customer_transactions(customer_id)
        
        if not transactions:
            logger.warning(f"No transactions found for customer {customer_id}")
            return CustomerRiskProfile(
                customer_id=customer_id,
                last_updated=datetime.now().isoformat()
            )
        
        # Calculate individual risk scores
        velocity_score, velocity_flags = self.calculate_velocity_score(transactions)
        location_score, location_flags = self.calculate_location_score(transactions)
        amount_score, amount_flags = self.calculate_amount_score(transactions)
        merchant_score, merchant_flags = self.calculate_merchant_score(transactions)
        behavioral_score, behavioral_flags = self.calculate_behavioral_score(transactions)
        ip_score, ip_flags = self.calculate_ip_score(transactions)
        
        # Calculate weighted scores
        credit_score = (
            velocity_score * 0.3 +
            amount_score * 0.4 +
            merchant_score * 0.2 +
            behavioral_score * 0.1
        )
        
        aml_score = (
            velocity_score * 0.2 +
            location_score * 0.3 +
            amount_score * 0.2 +
            merchant_score * 0.2 +
            ip_score * 0.1
        )
        
        combined_score = (credit_score + aml_score) / 2
        
        # Determine risk level
        risk_level = "LOW"
        for level, threshold in sorted(self.thresholds.items(), key=lambda x: x[1]):
            if combined_score >= threshold:
                risk_level = level
        
        # Collect all flags
        all_flags = (velocity_flags + location_flags + amount_flags + 
                    merchant_flags + behavioral_flags + ip_flags)
        
        # Calculate summary statistics
        amounts = [tx['amount'] for tx in transactions]
        merchants = set(tx.get('merchant_name', 'Unknown') for tx in transactions)
        locations = set((tx['location_lat'], tx['location_lon']) for tx in transactions)
        ips = set(tx.get('ip_address', 'Unknown') for tx in transactions)
        
        profile = CustomerRiskProfile(
            customer_id=customer_id,
            credit_score=round(credit_score, 3),
            aml_score=round(aml_score, 3),
            combined_score=round(combined_score, 3),
            risk_level=risk_level,
            last_updated=datetime.now().isoformat(),
            total_transactions=len(transactions),
            total_amount=round(sum(amounts), 2),
            avg_transaction_amount=round(statistics.mean(amounts), 2),
            unique_merchants=len(merchants),
            unique_locations=len(locations),
            unique_ips=len(ips),
            velocity_flags=len(velocity_flags),
            location_anomalies=len(location_flags),
            amount_anomalies=len(amount_flags),
            merchant_risk_flags=len(merchant_flags),
            behavioral_flags=len(behavioral_flags),
            fraud_indicators=all_flags
        )
        
        logger.info(f"Risk profile calculated for {customer_id}: {risk_level} ({combined_score:.3f})")
        return profile
    
    def store_risk_profile(self, profile: CustomerRiskProfile):
        """Store customer risk profile in Redis"""
        try:
            key = f"{self.key_prefix}:risk_profile:{profile.customer_id}"
            
            # Convert profile to dict and store
            profile_data = asdict(profile)
            profile_data['fraud_indicators'] = json.dumps(profile.fraud_indicators)
            
            # Store in Redis hash
            self.redis_client.hset(key, mapping=profile_data)
            
            # Set TTL
            self.redis_client.expire(key, self.config.transaction_ttl)
            
            # Also store in sorted set for quick retrieval of high-risk customers
            risk_key = f"{self.key_prefix}:risk_rankings"
            self.redis_client.zadd(risk_key, {profile.customer_id: profile.combined_score})
            
            logger.info(f"Stored risk profile for {profile.customer_id}")
            
        except Exception as e:
            logger.error(f"Error storing risk profile for {profile.customer_id}: {e}")
    
    def get_risk_profile(self, customer_id: str) -> Optional[CustomerRiskProfile]:
        """Retrieve customer risk profile from Redis"""
        try:
            key = f"{self.key_prefix}:risk_profile:{customer_id}"
            profile_data = self.redis_client.hgetall(key)
            
            if not profile_data:
                return None
                
            # Convert bytes to strings
            profile_dict = {}
            for k, v in profile_data.items():
                if isinstance(k, bytes):
                    k = k.decode('utf-8')
                if isinstance(v, bytes):
                    v = v.decode('utf-8')
                profile_dict[k] = v
            
            # Convert data types
            profile_dict['credit_score'] = float(profile_dict['credit_score'])
            profile_dict['aml_score'] = float(profile_dict['aml_score'])
            profile_dict['combined_score'] = float(profile_dict['combined_score'])
            profile_dict['total_transactions'] = int(profile_dict['total_transactions'])
            profile_dict['total_amount'] = float(profile_dict['total_amount'])
            profile_dict['avg_transaction_amount'] = float(profile_dict['avg_transaction_amount'])
            profile_dict['unique_merchants'] = int(profile_dict['unique_merchants'])
            profile_dict['unique_locations'] = int(profile_dict['unique_locations'])
            profile_dict['unique_ips'] = int(profile_dict['unique_ips'])
            profile_dict['velocity_flags'] = int(profile_dict['velocity_flags'])
            profile_dict['location_anomalies'] = int(profile_dict['location_anomalies'])
            profile_dict['amount_anomalies'] = int(profile_dict['amount_anomalies'])
            profile_dict['merchant_risk_flags'] = int(profile_dict['merchant_risk_flags'])
            profile_dict['behavioral_flags'] = int(profile_dict['behavioral_flags'])
            profile_dict['fraud_indicators'] = json.loads(profile_dict.get('fraud_indicators', '[]'))
            
            return CustomerRiskProfile(**profile_dict)
            
        except Exception as e:
            logger.error(f"Error retrieving risk profile for {customer_id}: {e}")
            return None
    
    def get_top_risky_customers(self, limit: int = 50) -> List[Tuple[str, float]]:
        """Get top risky customers by combined score"""
        try:
            risk_key = f"{self.key_prefix}:risk_rankings"
            # Get customers with highest scores (ZREVRANGE for descending order)
            results = self.redis_client.zrevrange(risk_key, 0, limit-1, withscores=True)
            
            return [(customer_id.decode('utf-8') if isinstance(customer_id, bytes) else customer_id, 
                    score) for customer_id, score in results]
            
        except Exception as e:
            logger.error(f"Error getting top risky customers: {e}")
            return []
    
    def analyze_all_customers(self):
        """Analyze all customers and update their risk profiles"""
        try:
            logger.info("Starting comprehensive customer risk analysis...")
            
            # Get all unique customer IDs - improved method
            pattern = f"{self.key_prefix}:transaction:*"
            transaction_keys = self.redis_client.keys(pattern)
            
            logger.info(f"Found {len(transaction_keys)} transaction keys to analyze")
            
            customer_ids = set()
            processed_keys = 0
            
            # Use batching to process keys efficiently
            for key in transaction_keys:
                try:
                    processed_keys += 1
                    if processed_keys % 1000 == 0:
                        logger.info(f"Processed {processed_keys}/{len(transaction_keys)} keys...")
                    
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                    
                    # Try hash first
                    customer_id = self.redis_client.hget(key, 'customer_id')
                    
                    if not customer_id:
                        # Try as JSON string
                        raw_data = self.redis_client.get(key)
                        if raw_data:
                            try:
                                raw_str = raw_data.decode('utf-8') if isinstance(raw_data, bytes) else str(raw_data)
                                json_data = json.loads(raw_str)
                                customer_id = json_data.get('customer_id')
                            except (json.JSONDecodeError, AttributeError):
                                continue
                    else:
                        customer_id = customer_id.decode('utf-8') if isinstance(customer_id, bytes) else str(customer_id)
                    
                    if customer_id:
                        customer_ids.add(customer_id)
                        
                except Exception as e:
                    logger.debug(f"Error processing key {key}: {e}")
                    continue
            
            logger.info(f"Found {len(customer_ids)} unique customers to analyze")
            
            if len(customer_ids) == 0:
                logger.warning("No customer IDs found! Check your Redis data structure.")
                return
            
            # Analyze each customer with progress tracking
            analyzed = 0
            errors = 0
            
            for customer_id in customer_ids:
                try:
                    profile = self.calculate_customer_risk_score(customer_id)
                    if profile.total_transactions > 0:  # Only store if we found transactions
                        self.store_risk_profile(profile)
                        analyzed += 1
                    else:
                        logger.debug(f"No transactions found for customer {customer_id}")
                    
                    if analyzed % 10 == 0 and analyzed > 0:
                        logger.info(f"Analyzed {analyzed}/{len(customer_ids)} customers")
                        
                except Exception as e:
                    errors += 1
                    logger.error(f"Error analyzing customer {customer_id}: {e}")
                    if errors > 10:  # Stop if too many errors
                        logger.error("Too many errors, stopping analysis")
                        break
                    continue
            
            logger.info(f"Completed analysis of {analyzed} customers ({errors} errors)")
            
            if analyzed > 0:
                # Show top risky customers
                top_risky = self.get_top_risky_customers(10)
                logger.info("Top 10 risky customers:")
                for customer_id, score in top_risky:
                    logger.info(f"  {customer_id}: {score:.3f}")
            else:
                logger.warning("No customers were successfully analyzed!")
                
        except Exception as e:
            logger.error(f"Error in analyze_all_customers: {e}")
            import traceback
            logger.error(traceback.format_exc())

def main():
    """Main function to run the risk scoring agent"""
    print("🔍 Financial Risk Scoring Agent")
    print("=" * 50)
    
    # Load configurations
    redis_config = RedisConfig.from_env()
    agent_config = AgentConfig.from_env()
    
    # Initialize agent
    scoring_agent = RiskScoringAgent(redis_config, agent_config)
    
    try:
        # Test Redis connection
        scoring_agent.redis_client.ping()
        print("✅ Connected to Redis successfully")
        
        # Run analysis
        print("\n🚀 Starting risk analysis...")
        scoring_agent.analyze_all_customers()
        
        print("\n📊 Analysis complete! Risk profiles stored in Redis.")
        print("🔑 Redis keys created:")
        print(f"  - {agent_config.key_prefix}:risk_profile:{{customer_id}} (individual profiles)")
        print(f"  - {agent_config.key_prefix}:risk_rankings (sorted set of risk scores)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()