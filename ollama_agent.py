# Example usage and demonstration
def main():
    """
    Demonstration of the Unified Risk Detection System with Ollama LLM
    """
    # Initialize system with Ollama for local LLM inference
    system = UnifiedRiskDetectionSystem(
        redis_host='localhost',
        redis_port=6379,
        use_ollama=True  # Enable Ollama for AI explanations
    )
    
    # Example 1: Monitor specific customers from your Redis data
    print("=" * 80)
    print("🏦 UNIFIED FINANCIAL RISK DETECTION SYSTEM")
    print("Powered by Ollama LLM for Intelligent Risk Analysis")
    print("=" * 80)
    
    # Test with customers from your Redis database
    test_customers = [
        "CUST_C7E8A40F",  # From your first screenshot
        "CUST_E18E2572",  # From your second screenshot
    ]
    
    for customer_id in test_customers:
        print(f"\n{'='*80}")
        print(f"📊 CUSTOMER RISK ANALYSIS: {customer_id}")
        print("import redis")
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import requests
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import defaultdict
import warnings
import redis
warnings.filterwarnings('ignore')

# Configuration
class RiskLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

@dataclass
class TransactionProfile:
    """Customer transaction profile for analysis"""
    customer_id: str
    total_transactions: int
    avg_amount: float
    std_amount: float
    max_amount: float
    transaction_frequency: float  # transactions per day
    unique_merchants: int
    unique_locations: int
    risk_score_trend: float
    sudden_spike_count: int
    unusual_hours_count: int
    cross_border_ratio: float

class OllamaLLM:
    """Ollama LLM wrapper for local model inference"""
    
    def __init__(self, base_url="http://localhost:11434/api/chat", model="phi3:mini", temperature=0.2):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
    
    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial risk analyst providing clear, actionable risk assessments."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            print(f"[ERROR] Ollama LLM call failed: {e}")
            return None


class UnifiedRiskDetectionSystem:
    """
    Unified system for Credit Risk and AML Detection
    Integrates behavioral analysis, location-based risk, and pattern detection
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, use_ollama=True):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.use_ollama = use_ollama
        if use_ollama:
            self.llm = OllamaLLM()
        
        # Risk thresholds
        self.thresholds = {
            'velocity_increase': 1.5,  # 50% increase in transaction velocity
            'amount_spike': 2.0,  # 2x normal amount
            'rapid_transactions': 5,  # 5+ transactions in 1 hour
            'cross_border_threshold': 0.3,  # 30% cross-border transactions
            'unusual_hour_threshold': 0.2,  # 20% transactions in unusual hours
        }
        
        # Location risk scores (simplified - would be more comprehensive in production)
        self.location_risk_scores = {
            'high_risk_countries': ['North Korea', 'Iran', 'Syria', 'Cuba'],
            'medium_risk_countries': ['Russia', 'China', 'Pakistan', 'Nigeria'],
            'offshore_centers': ['Cayman Islands', 'British Virgin Islands', 'Panama']
        }
    
    def fetch_transaction_data(self, customer_id: str, days_back: int = 90) -> pd.DataFrame:
        """
        Fetch transaction data from Redis for a specific customer
        Adapted for the key pattern: financial:transaction:TXN_XXXXX
        """
        transactions = []
        
        # Scan for all transaction keys
        pattern = "financial:transaction:*"
        for key in self.redis_client.scan_iter(match=pattern, count=100):
            try:
                # Get the hash data
                data = self.redis_client.hgetall(key)
                if data and data.get('customer_id') == customer_id:
                    # Parse the nested JSON data field
                    if 'data' in data:
                        try:
                            nested_data = json.loads(data['data'])
                            # Merge nested data with main fields
                            transaction = {**data, **nested_data}
                        except:
                            transaction = data
                    else:
                        transaction = data
                    
                    transactions.append(transaction)
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                continue
        
        if not transactions:
            # If no data in Redis, return sample data for demonstration
            transactions = self._generate_sample_transactions(customer_id, days_back)
        
        df = pd.DataFrame(transactions)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Filter by days_back
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df = df[df['timestamp'] >= cutoff_date]
        
        return df
    
    def _generate_sample_transactions(self, customer_id: str, days: int) -> List[Dict]:
        """Generate sample transactions for demonstration"""
        np.random.seed(hash(customer_id) % 1000)
        transactions = []
        
        base_amount = np.random.uniform(100, 1000)
        for i in range(days * 3):  # ~3 transactions per day
            timestamp = datetime.now() - timedelta(days=days-i//3, hours=np.random.randint(0, 24))
            
            # Simulate different patterns
            if i > days * 2.5:  # Recent suspicious activity
                amount = base_amount * np.random.uniform(1.5, 3.0)
            else:
                amount = base_amount * np.random.uniform(0.8, 1.2)
            
            transactions.append({
                'transaction_id': f'TXN_{np.random.randint(100000, 999999)}',
                'customer_id': customer_id,
                'amount': round(amount, 2),
                'currency': np.random.choice(['USD', 'EUR', 'GBP'], p=[0.6, 0.3, 0.1]),
                'timestamp': timestamp.isoformat(),
                'location_country': np.random.choice(['US', 'UK', 'Germany', 'China', 'Nigeria'], 
                                                    p=[0.5, 0.2, 0.15, 0.1, 0.05]),
                'merchant_name': f'Merchant_{np.random.randint(1, 50)}',
                'risk_score': np.random.uniform(0, 0.5) if i < days * 2 else np.random.uniform(0.3, 0.8)
            })
        
        return transactions
    
    def analyze_transaction_patterns(self, df: pd.DataFrame) -> TransactionProfile:
        """
        Analyze transaction patterns for AML and fraud detection
        """
        if df.empty:
            return None
        
        customer_id = df['customer_id'].iloc[0]
        
        # Basic statistics
        total_transactions = len(df)
        avg_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        max_amount = df['amount'].max()
        
        # Time-based analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        date_range = (df['timestamp'].max() - df['timestamp'].min()).days + 1
        transaction_frequency = total_transactions / max(date_range, 1)
        
        # Unusual hours (midnight to 6 AM)
        df['hour'] = df['timestamp'].dt.hour
        unusual_hours_count = len(df[(df['hour'] >= 0) & (df['hour'] < 6)])
        
        # Velocity and spike detection
        df['amount_zscore'] = (df['amount'] - avg_amount) / (std_amount + 1e-10)
        sudden_spike_count = len(df[df['amount_zscore'] > 2])
        
        # Geographic analysis
        unique_locations = df['location_country'].nunique() if 'location_country' in df else 1
        home_country = df['location_country'].mode()[0] if 'location_country' in df else 'US'
        cross_border_ratio = 1 - (df['location_country'] == home_country).mean() if 'location_country' in df else 0
        
        # Merchant diversity
        unique_merchants = df['merchant_name'].nunique() if 'merchant_name' in df else 1
        
        # Risk score trend
        if 'risk_score' in df:
            recent_risk = df.tail(10)['risk_score'].mean()
            historical_risk = df.head(len(df)-10)['risk_score'].mean() if len(df) > 10 else recent_risk
            risk_score_trend = recent_risk - historical_risk
        else:
            risk_score_trend = 0
        
        return TransactionProfile(
            customer_id=customer_id,
            total_transactions=total_transactions,
            avg_amount=avg_amount,
            std_amount=std_amount,
            max_amount=max_amount,
            transaction_frequency=transaction_frequency,
            unique_merchants=unique_merchants,
            unique_locations=unique_locations,
            risk_score_trend=risk_score_trend,
            sudden_spike_count=sudden_spike_count,
            unusual_hours_count=unusual_hours_count,
            cross_border_ratio=cross_border_ratio
        )
    
    def detect_aml_patterns(self, df: pd.DataFrame, profile: TransactionProfile) -> Dict:
        """
        Detect specific AML patterns like structuring, layering, and integration
        """
        aml_flags = {
            'structuring': False,
            'rapid_movement': False,
            'unusual_pattern': False,
            'geographic_risk': False,
            'velocity_change': False
        }
        
        if df.empty:
            return aml_flags
        
        # Structuring detection (multiple transactions just below reporting threshold)
        reporting_threshold = 10000
        df['date'] = df['timestamp'].dt.date
        daily_totals = df.groupby('date')['amount'].sum()
        suspicious_days = daily_totals[(daily_totals > reporting_threshold * 0.8) & 
                                       (daily_totals < reporting_threshold)]
        if len(suspicious_days) > 2:
            aml_flags['structuring'] = True
        
        # Rapid movement detection
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600  # hours
        rapid_transactions = df[df['time_diff'] < 1]  # Within 1 hour
        if len(rapid_transactions) > 3:
            aml_flags['rapid_movement'] = True
        
        # Unusual pattern detection
        if profile.sudden_spike_count > 3 or profile.unusual_hours_count / len(df) > 0.2:
            aml_flags['unusual_pattern'] = True
        
        # Geographic risk
        if 'location_country' in df:
            high_risk_transactions = df[df['location_country'].isin(
                self.location_risk_scores['high_risk_countries'] + 
                self.location_risk_scores['offshore_centers']
            )]
            if len(high_risk_transactions) > 0:
                aml_flags['geographic_risk'] = True
        
        # Velocity change detection
        if len(df) > 20:
            recent_freq = len(df.tail(10)) / 10
            historical_freq = len(df.head(len(df)-10)) / (len(df)-10)
            if recent_freq > historical_freq * 1.5:
                aml_flags['velocity_change'] = True
        
        return aml_flags
    
    def calculate_risk_score(self, profile: TransactionProfile, aml_flags: Dict, df: pd.DataFrame) -> Tuple[int, float, float, float]:
        """
        Calculate unified risk score (1-5) with separate credit and AML scores
        """
        # Credit Risk Score (0-100)
        credit_risk_points = 0
        
        # Transaction behavior for credit risk
        if profile.transaction_frequency > 10:
            credit_risk_points += 15
        if profile.std_amount > profile.avg_amount * 0.5:  # High variance
            credit_risk_points += 10
        if profile.max_amount > profile.avg_amount * 5:  # Very large transactions
            credit_risk_points += 15
        if profile.unique_merchants > 30:  # Too many merchants
            credit_risk_points += 10
        if profile.risk_score_trend > 0.3:  # Increasing risk trend
            credit_risk_points += 20
        
        # AML Risk Score (0-100)
        aml_risk_points = 0
        
        # AML specific patterns
        if aml_flags['structuring']:
            aml_risk_points += 25
        if aml_flags['rapid_movement']:
            aml_risk_points += 20
        if aml_flags['unusual_pattern']:
            aml_risk_points += 10
        if aml_flags['geographic_risk']:
            aml_risk_points += 20
        if aml_flags['velocity_change']:
            aml_risk_points += 15
        
        # Geographic and timing risks
        if profile.cross_border_ratio > 0.5:
            aml_risk_points += 10
        if profile.unusual_hours_count / profile.total_transactions > 0.3:
            aml_risk_points += 10
        
        # Calculate combined score (weighted average)
        credit_weight = 0.4
        aml_weight = 0.6
        combined_score = (credit_risk_points * credit_weight + aml_risk_points * aml_weight)
        
        # Map to 1-5 scale
        if combined_score < 20:
            risk_level = 1
        elif combined_score < 40:
            risk_level = 2
        elif combined_score < 60:
            risk_level = 3
        elif combined_score < 80:
            risk_level = 4
        else:
            risk_level = 5
        
        return risk_level, combined_score, credit_risk_points, aml_risk_points
    
    def generate_risk_explanation(self, customer_id: str, risk_level: int, 
                                 profile: TransactionProfile, aml_flags: Dict, 
                                 credit_risk_score: float, aml_risk_score: float) -> Tuple[str, List[str]]:
        """
        Use Ollama to generate detailed risk explanation with 3-point summary
        """
        if self.use_ollama and self.llm:
            try:
                # Prepare context for Ollama
                context = f"""
                Analyze this customer's financial risk profile and provide a detailed explanation.
                
                Customer ID: {customer_id}
                Overall Risk Level: {risk_level}/5
                Credit Risk Score: {credit_risk_score:.1f}%
                AML Risk Score: {aml_risk_score:.1f}%
                
                Transaction Profile:
                - Total Transactions: {profile.total_transactions}
                - Average Amount: ${profile.avg_amount:.2f}
                - Standard Deviation: ${profile.std_amount:.2f}
                - Maximum Amount: ${profile.max_amount:.2f}
                - Transaction Frequency: {profile.transaction_frequency:.2f} per day
                - Sudden Spikes: {profile.sudden_spike_count} transactions above 2x normal
                - Cross-border Ratio: {profile.cross_border_ratio:.2%}
                - Unique Merchants: {profile.unique_merchants}
                - Unique Locations: {profile.unique_locations}
                - Unusual Hours Activity: {profile.unusual_hours_count} transactions
                
                AML Red Flags Detected:
                - Structuring Pattern: {aml_flags['structuring']}
                - Rapid Fund Movement: {aml_flags['rapid_movement']}
                - Unusual Pattern: {aml_flags['unusual_pattern']}
                - Geographic Risk: {aml_flags['geographic_risk']}
                - Velocity Change: {aml_flags['velocity_change']}
                
                Based on this information:
                1. Explain WHY this customer is classified as risk level {risk_level}
                2. Identify the most concerning patterns or behaviors
                3. Provide exactly 3 key summary points (numbered list)
                4. Recommend specific actions for the compliance team
                
                Format your response with clear sections:
                - Risk Explanation
                - Key Summary Points (exactly 3)
                - Recommended Actions
                """
                
                response = self.llm.generate(context)
                
                if response:
                    # Extract 3-point summary from response
                    summary_points = self._extract_summary_points(response)
                    return response, summary_points
                    
            except Exception as e:
                print(f"Ollama generation failed: {e}")
        
        # Fallback to rule-based explanation
        explanation, summary = self._generate_rule_based_explanation(
            risk_level, profile, aml_flags, credit_risk_score, aml_risk_score
        )
        return explanation, summary
    
    def _extract_summary_points(self, text: str) -> List[str]:
        """Extract 3 summary points from the LLM response"""
        lines = text.split('\n')
        summary_points = []
        in_summary = False
        
        for line in lines:
            if 'summary points' in line.lower() or 'key summary' in line.lower():
                in_summary = True
                continue
            if in_summary and line.strip():
                # Look for numbered points
                if line.strip()[0].isdigit() and '.' in line[:3]:
                    summary_points.append(line.strip())
                    if len(summary_points) == 3:
                        break
        
        # If extraction failed, create default points
        if len(summary_points) < 3:
            summary_points = [
                "1. Customer shows elevated risk patterns requiring review",
                "2. Multiple risk indicators detected in transaction history",
                "3. Enhanced monitoring recommended"
            ]
        
        return summary_points
    
    def _generate_rule_based_explanation(self, risk_level: int, 
                                        profile: TransactionProfile, aml_flags: Dict,
                                        credit_risk_score: float, aml_risk_score: float) -> Tuple[str, List[str]]:
        """
        Generate rule-based explanation when Ollama is not available
        """
        risk_name = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}[risk_level]
        
        explanation = f"**Risk Assessment: {risk_name} (Level {risk_level}/5)**\n\n"
        explanation += f"**Credit Risk Score: {credit_risk_score:.1f}% | AML Risk Score: {aml_risk_score:.1f}%**\n\n"
        
        # Key findings
        explanation += "**Risk Explanation:**\n"
        
        summary_points = []
        
        if risk_level >= 4:
            if aml_flags['structuring']:
                explanation += "• ⚠️ STRUCTURING DETECTED: Multiple transactions just below reporting threshold indicate potential money laundering\n"
                summary_points.append("1. Structuring pattern detected - possible money laundering attempt")
            if aml_flags['rapid_movement']:
                explanation += "• ⚠️ RAPID FUND MOVEMENT: Multiple transactions within short time periods suggest layering activity\n"
                if len(summary_points) < 3:
                    summary_points.append(f"{len(summary_points)+1}. Rapid fund movement indicates potential layering")
            if profile.sudden_spike_count > 5:
                explanation += f"• ⚠️ UNUSUAL ACTIVITY: {profile.sudden_spike_count} transactions significantly above normal amounts\n"
                if len(summary_points) < 3:
                    summary_points.append(f"{len(summary_points)+1}. {profile.sudden_spike_count} suspicious transaction spikes detected")
        
        if risk_level >= 3:
            if aml_flags['geographic_risk']:
                explanation += "• 🌍 GEOGRAPHIC RISK: Transactions involving high-risk or sanctioned jurisdictions\n"
                if len(summary_points) < 3:
                    summary_points.append(f"{len(summary_points)+1}. High-risk geographic activity detected")
            if profile.cross_border_ratio > 0.3:
                explanation += f"• 🌍 CROSS-BORDER ACTIVITY: {profile.cross_border_ratio:.1%} of transactions are international\n"
                if len(summary_points) < 3:
                    summary_points.append(f"{len(summary_points)+1}. Elevated cross-border transaction ratio ({profile.cross_border_ratio:.1%})")
        
        # Fill remaining summary points if needed
        while len(summary_points) < 3:
            if profile.transaction_frequency > 5:
                summary_points.append(f"{len(summary_points)+1}. High transaction frequency ({profile.transaction_frequency:.1f}/day)")
            elif profile.unique_merchants > 20:
                summary_points.append(f"{len(summary_points)+1}. Unusually diverse merchant activity ({profile.unique_merchants} merchants)")
            else:
                summary_points.append(f"{len(summary_points)+1}. Overall risk score indicates need for review")
        
        # Transaction patterns
        explanation += f"\n**Transaction Behavior:**\n"
        explanation += f"• Average transaction: ${profile.avg_amount:.2f}\n"
        explanation += f"• Transaction frequency: {profile.transaction_frequency:.1f} per day\n"
        explanation += f"• Unique merchants: {profile.unique_merchants}\n"
        explanation += f"• Unique locations: {profile.unique_locations}\n"
        
        # Key Summary Points
        explanation += "\n**Key Summary Points:**\n"
        for point in summary_points[:3]:
            explanation += f"{point}\n"
        
        # Recommendations
        explanation += "\n**Recommended Actions:**\n"
        if risk_level >= 4:
            explanation += "• 🔴 IMMEDIATE ACTION: File SAR (Suspicious Activity Report)\n"
            explanation += "• 🔴 Freeze account pending investigation\n"
            explanation += "• 🔴 Enhanced due diligence (EDD) required\n"
            explanation += "• 🔴 Real-time transaction monitoring activated\n"
        elif risk_level == 3:
            explanation += "• 🟡 Schedule for enhanced review within 24 hours\n"
            explanation += "• 🟡 Request additional KYC documentation\n"
            explanation += "• 🟡 Set up automated alerts for unusual patterns\n"
            explanation += "• 🟡 Monitor for pattern escalation\n"
        else:
            explanation += "• 🟢 Continue standard monitoring protocols\n"
            explanation += "• 🟢 Quarterly review cycle appropriate\n"
            explanation += "• 🟢 No immediate action required\n"
        
        return explanation, summary_points
    
    def monitor_customer(self, customer_id: str) -> Dict:
        """
        Main monitoring function that combines all analysis
        """
        print(f"[INFO] Starting analysis for customer: {customer_id}")
        
        # Fetch data
        df = self.fetch_transaction_data(customer_id)
        
        if df.empty:
            return {
                'customer_id': customer_id,
                'risk_level': 1,
                'combined_score': 0,
                'credit_risk_score': 0,
                'aml_risk_score': 0,
                'explanation': 'No transaction data available',
                'summary_points': [],
                'aml_flags': {},
                'profile': None
            }
        
        print(f"[INFO] Found {len(df)} transactions for analysis")
        
        # Analyze patterns
        profile = self.analyze_transaction_patterns(df)
        aml_flags = self.detect_aml_patterns(df, profile)
        
        # Calculate risk with separate scores
        risk_level, combined_score, credit_score, aml_score = self.calculate_risk_score(profile, aml_flags, df)
        
        print(f"[INFO] Risk Scores - Credit: {credit_score:.1f}%, AML: {aml_score:.1f}%, Combined: {combined_score:.1f}%")
        
        # Generate explanation with 3-point summary
        explanation, summary_points = self.generate_risk_explanation(
            customer_id, risk_level, profile, aml_flags, credit_score, aml_score
        )
        
        # Store results back to Redis
        result = {
            'customer_id': customer_id,
            'risk_level': risk_level,
            'combined_score': combined_score,
            'credit_risk_score': credit_score,
            'aml_risk_score': aml_score,
            'explanation': explanation,
            'summary_points': summary_points,
            'aml_flags': aml_flags,
            'profile': profile.__dict__ if profile else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to Redis with risk profile
        risk_profile_key = f"risk:profile:{customer_id}"
        self.redis_client.hset(risk_profile_key, mapping={
            'risk_level': risk_level,
            'credit_risk_score': credit_score,
            'aml_risk_score': aml_score,
            'combined_score': combined_score,
            'last_analysis': datetime.now().isoformat(),
            'summary': json.dumps(summary_points),
            'aml_flags': json.dumps(aml_flags)
        })
        
        # Set expiry for risk profile (24 hours)
        self.redis_client.expire(risk_profile_key, 86400)
        
        return result
    
    def batch_monitor(self, customer_ids: List[str]) -> pd.DataFrame:
        """
        Monitor multiple customers and return consolidated results
        """
        results = []
        for customer_id in customer_ids:
            result = self.monitor_customer(customer_id)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def real_time_alert(self, transaction: Dict) -> Optional[Dict]:
        """
        Real-time transaction monitoring for immediate alerts
        """
        customer_id = transaction.get('customer_id')
        if not customer_id:
            return None
        
        # Get recent history
        df = self.fetch_transaction_data(customer_id, days_back=30)
        
        if df.empty:
            return None
        
        # Check for immediate red flags
        avg_amount = df['amount'].mean()
        current_amount = transaction.get('amount', 0)
        
        alert = None
        if current_amount > avg_amount * 3:  # 3x normal amount
            alert = {
                'type': 'AMOUNT_SPIKE',
                'severity': 'HIGH',
                'message': f'Transaction amount ${current_amount:.2f} is {current_amount/avg_amount:.1f}x normal',
                'customer_id': customer_id,
                'transaction_id': transaction.get('transaction_id')
            }
        
        # Check velocity
        recent_transactions = df[df['timestamp'] > datetime.now() - timedelta(hours=1)]
        if len(recent_transactions) > 5:
            alert = {
                'type': 'VELOCITY',
                'severity': 'MEDIUM',
                'message': f'{len(recent_transactions)} transactions in the last hour',
                'customer_id': customer_id,
                'transaction_id': transaction.get('transaction_id')
            }
        
        if alert:
            # Store alert in Redis
            self.redis_client.lpush(f"alerts:queue", json.dumps(alert))
            
        return alert


# Example usage and demonstration
def main():
    """
    Demonstration of the Unified Risk Detection System
    """
    # Initialize system (set your OpenAI API key if available)
    system = UnifiedRiskDetectionSystem(
        redis_host='localhost',
        redis_port=6379,
        openai_api_key=None  # Set your OpenAI API key here
    )
    
    # Example 1: Monitor a specific customer
    print("=" * 80)
    print("UNIFIED RISK DETECTION SYSTEM - DEMONSTRATION")
    print("=" * 80)
    
    # Using the customer ID from your Redis data
    customer_id = "CUST_C7E8A40F"
    
    print(f"\n📊 Analyzing Customer: {customer_id}")
    print("-" * 40)
    
    result = system.monitor_customer(customer_id)
    
    print(f"Risk Level: {result['risk_level']}/5")
    print(f"Risk Percentage: {result['risk_percentage']:.1f}%")
    print(f"\nAML Flags Detected:")
    for flag, value in result['aml_flags'].items():
        if value:
            print(f"  • {flag}: ✓")
    
    print(f"\n{result['explanation']}")
    
    # Example 2: Real-time transaction monitoring
    print("\n" + "=" * 80)
    print("REAL-TIME TRANSACTION MONITORING")
    print("=" * 80)
    
    # Simulate a new transaction
    new_transaction = {
        'transaction_id': 'TXN_NEW001',
        'customer_id': customer_id,
        'amount': 5000.00,  # Suspicious amount
        'timestamp': datetime.now().isoformat(),
        'location_country': 'Nigeria',  # High-risk country
        'merchant_name': 'Unknown Merchant'
    }
    
    alert = system.real_time_alert(new_transaction)
    if alert:
        print(f"\n🚨 ALERT TRIGGERED!")
        print(f"Type: {alert['type']}")
        print(f"Severity: {alert['severity']}")
        print(f"Message: {alert['message']}")
    else:
        print("\n✅ Transaction passed real-time checks")
    
    # Example 3: Batch monitoring
    print("\n" + "=" * 80)
    print("BATCH CUSTOMER MONITORING")
    print("=" * 80)
    
    customer_list = [
        "CUST_C7E8A40F",
        "CUST_A1B2C3D4",
        "CUST_E5F6G7H8"
    ]
    
    batch_results = system.batch_monitor(customer_list)
    
    print("\nRisk Summary:")
    print(batch_results[['customer_id', 'risk_level', 'risk_percentage']].to_string(index=False))
    
    # Get high-risk customers
    high_risk = batch_results[batch_results['risk_level'] >= 4]
    if not high_risk.empty:
        print(f"\n⚠️ High Risk Customers Requiring Immediate Attention: {len(high_risk)}")
        for _, customer in high_risk.iterrows():
            print(f"  • {customer['customer_id']}: Level {customer['risk_level']}")


if __name__ == "__main__":
    main()