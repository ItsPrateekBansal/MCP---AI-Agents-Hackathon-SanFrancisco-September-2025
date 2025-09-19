# risk_analysis_demo.py - Demo and testing script for Risk Scoring Agent
import redis
import json
from datetime import datetime
from redis_config import RedisConfig, AgentConfig
from scoring_agent import RiskScoringAgent, CustomerRiskProfile

def print_customer_analysis(agent: RiskScoringAgent, customer_id: str):
    """Print detailed analysis for a specific customer"""
    print(f"\n{'='*60}")
    print(f"📊 DETAILED RISK ANALYSIS FOR CUSTOMER: {customer_id}")
    print(f"{'='*60}")
    
    # Get risk profile
    profile = agent.get_risk_profile(customer_id)
    if not profile:
        print(f"❌ No risk profile found for {customer_id}")
        return
    
    # Display overview
    print(f"\n🎯 RISK OVERVIEW")
    print(f"   Combined Score: {profile.combined_score:.3f}")
    print(f"   Risk Level: {profile.risk_level}")
    print(f"   Credit Score: {profile.credit_score:.3f}")
    print(f"   AML Score: {profile.aml_score:.3f}")
    print(f"   Last Updated: {profile.last_updated}")
    
    # Transaction summary
    print(f"\n📈 TRANSACTION SUMMARY")
    print(f"   Total Transactions: {profile.total_transactions}")
    print(f"   Total Amount: ${profile.total_amount:,.2f}")
    print(f"   Average Amount: ${profile.avg_transaction_amount:,.2f}")
    print(f"   Unique Merchants: {profile.unique_merchants}")
    print(f"   Unique Locations: {profile.unique_locations}")
    print(f"   Unique IP Addresses: {profile.unique_ips}")
    
    # Risk flags breakdown
    print(f"\n🚩 RISK FLAGS BREAKDOWN")
    print(f"   Velocity Flags: {profile.velocity_flags}")
    print(f"   Location Anomalies: {profile.location_anomalies}")
    print(f"   Amount Anomalies: {profile.amount_anomalies}")
    print(f"   Merchant Risk Flags: {profile.merchant_risk_flags}")
    print(f"   Behavioral Flags: {profile.behavioral_flags}")
    
    # Detailed fraud indicators
    if profile.fraud_indicators:
        print(f"\n🔍 DETAILED FRAUD INDICATORS")
        for i, indicator in enumerate(profile.fraud_indicators, 1):
            print(f"   {i}. {indicator}")
    else:
        print(f"\n✅ No specific fraud indicators detected")

def print_top_risky_customers(agent: RiskScoringAgent, limit: int = 10):
    """Print top risky customers"""
    print(f"\n{'='*60}")
    print(f"🚨 TOP {limit} HIGHEST RISK CUSTOMERS")
    print(f"{'='*60}")
    
    top_risky = agent.get_top_risky_customers(limit)
    
    if not top_risky:
        print("❌ No risky customers found")
        return
    
    print(f"{'Rank':<6} {'Customer ID':<20} {'Risk Score':<12} {'Risk Level':<12}")
    print("-" * 60)
    
    for i, (customer_id, score) in enumerate(top_risky, 1):
        # Get full profile for risk level
        profile = agent.get_risk_profile(customer_id)
        risk_level = profile.risk_level if profile else "Unknown"
        
        # Color coding for risk levels
        if score >= 0.8:
            emoji = "🔴"
        elif score >= 0.6:
            emoji = "🟠"
        elif score >= 0.3:
            emoji = "🟡"
        else:
            emoji = "🟢"
            
        print(f"{emoji} {i:<4} {customer_id:<20} {score:<12.3f} {risk_level:<12}")

def print_risk_statistics(agent: RiskScoringAgent):
    """Print overall risk statistics"""
    print(f"\n{'='*60}")
    print(f"📊 OVERALL RISK STATISTICS")
    print(f"{'='*60}")
    
    # Get all customers
    all_risky = agent.get_top_risky_customers(1000)  # Get more to calculate stats
    
    if not all_risky:
        print("❌ No customer data found")
        return
    
    # Calculate statistics
    scores = [score for _, score in all_risky]
    risk_levels = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    
    for customer_id, score in all_risky:
        profile = agent.get_risk_profile(customer_id)
        if profile:
            risk_levels[profile.risk_level] += 1
    
    total_customers = len(all_risky)
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    
    print(f"\n📈 SCORE DISTRIBUTION")
    print(f"   Total Customers Analyzed: {total_customers}")
    print(f"   Average Risk Score: {avg_score:.3f}")
    print(f"   Highest Risk Score: {max_score:.3f}")
    print(f"   Lowest Risk Score: {min_score:.3f}")
    
    print(f"\n🎯 RISK LEVEL BREAKDOWN")
    for level, count in risk_levels.items():
        percentage = (count / total_customers * 100) if total_customers > 0 else 0
        if level == "CRITICAL":
            emoji = "🔴"
        elif level == "HIGH":
            emoji = "🟠"
        elif level == "MEDIUM":
            emoji = "🟡"
        else:
            emoji = "🟢"
        print(f"   {emoji} {level}: {count} customers ({percentage:.1f}%)")

def analyze_specific_customer(agent: RiskScoringAgent, customer_id: str):
    """Analyze a specific customer and show their transactions"""
    print(f"\n{'='*80}")
    print(f"🔍 DEEP DIVE ANALYSIS FOR CUSTOMER: {customer_id}")
    print(f"{'='*80}")
    
    # Get transactions
    transactions = agent.get_customer_transactions(customer_id)
    
    if not transactions:
        print(f"❌ No transactions found for customer {customer_id}")
        return
    
    print(f"\n📋 RECENT TRANSACTIONS ({len(transactions)} total)")
    print("-" * 120)
    print(f"{'Date':<12} {'Amount':<10} {'Merchant':<25} {'Location':<15} {'Device':<10} {'Risk':<6}")
    print("-" * 120)
    
    # Sort by timestamp
    sorted_txs = sorted(transactions, key=lambda x: x['timestamp'], reverse=True)
    
    for tx in sorted_txs[:10]:  # Show last 10 transactions
        date = tx['timestamp'][:10]
        amount = f"${tx['amount']:.2f}"
        merchant = tx.get('merchant_name', 'Unknown')[:24]
        location = tx.get('location_city', 'Unknown')[:14]
        device = tx.get('device_type', 'Unknown')[:9]
        risk = f"{tx['risk_score']:.2f}"
        
        print(f"{date:<12} {amount:<10} {merchant:<25} {location:<15} {device:<10} {risk:<6}")
    
    if len(transactions) > 10:
        print(f"... and {len(transactions) - 10} more transactions")
    
    # Now show the risk analysis
    print_customer_analysis(agent, customer_id)

def interactive_demo(agent: RiskScoringAgent):
    """Interactive demo mode"""
    print(f"\n{'='*60}")
    print("🎮 INTERACTIVE RISK ANALYSIS MODE")
    print(f"{'='*60}")
    print("Available commands:")
    print("  1. 'top' - Show top risky customers")
    print("  2. 'stats' - Show overall statistics")
    print("  3. 'analyze <customer_id>' - Analyze specific customer")
    print("  4. 'refresh' - Re-analyze all customers")
    print("  5. 'quit' - Exit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'quit':
                print("👋 Goodbye!")
                break
            elif command == 'top':
                print_top_risky_customers(agent, 15)
            elif command == 'stats':
                print_risk_statistics(agent)
            elif command == 'refresh':
                print("🔄 Re-analyzing all customers...")
                agent.analyze_all_customers()
                print("✅ Analysis complete!")
            elif command.startswith('analyze '):
                customer_id = command.split(' ', 1)[1].strip()
                analyze_specific_customer(agent, customer_id)
            else:
                print("❌ Unknown command. Try 'top', 'stats', 'analyze <customer_id>', 'refresh', or 'quit'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_sample_analysis(agent: RiskScoringAgent):
    """Run a sample analysis to demonstrate the system"""
    print("🚀 Running sample analysis...")
    
    # Get some sample customers
    top_risky = agent.get_top_risky_customers(5)
    
    if not top_risky:
        print("❌ No customer data found. Make sure to run the data generation agent first.")
        return
    
    # Show top risky customers
    print_top_risky_customers(agent, 10)
    
    # Show overall statistics
    print_risk_statistics(agent)
    
    # Analyze the most risky customer in detail
    if top_risky:
        most_risky_customer = top_risky[0][0]
        analyze_specific_customer(agent, most_risky_customer)

def main():
    """Main demo function"""
    print("🔍 Financial Risk Scoring Agent - Demo & Analysis")
    print("=" * 60)
    
    # Load configurations
    redis_config = RedisConfig.from_env()
    agent_config = AgentConfig.from_env()
    
    # Initialize agent
    try:
        agent = RiskScoringAgent(redis_config, agent_config)
        agent.redis_client.ping()
        print("✅ Connected to Redis successfully")
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        print("📝 Make sure Redis is running and .env file is configured correctly")
        return False
    
    print("\nSelect demo mode:")
    print("1. Run sample analysis (recommended first time)")
    print("2. Interactive mode")
    print("3. Re-analyze all customers")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            run_sample_analysis(agent)
        elif choice == '2':
            interactive_demo(agent)
        elif choice == '3':
            print("🔄 Re-analyzing all customers...")
            agent.analyze_all_customers()
            print("✅ Analysis complete!")
            run_sample_analysis(agent)
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
    
    return True

if __name__ == "__main__":
    main()