# debug_redis_data.py - Debug script to examine Redis data structure
import redis
import json
from redis_config import RedisConfig

def debug_redis_data():
    """Debug Redis data structure and content"""
    print("🔍 Redis Data Structure Debug")
    print("=" * 50)
    
    # Connect to Redis
    config = RedisConfig.from_env()
    r = redis.Redis(**config.to_dict())
    
    try:
        r.ping()
        print("✅ Connected to Redis successfully")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return
    
    # Get all keys
    print("\n📋 Examining Redis keys...")
    all_keys = r.keys("*")
    print(f"Total keys found: {len(all_keys)}")
    
    # Group keys by pattern
    key_patterns = {}
    for key in all_keys:
        key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
        
        # Extract pattern
        if ':' in key_str:
            pattern = ':'.join(key_str.split(':')[:-1]) + ':*'
        else:
            pattern = 'no_colon'
            
        if pattern not in key_patterns:
            key_patterns[pattern] = []
        key_patterns[pattern].append(key_str)
    
    print("\n📊 Key patterns found:")
    for pattern, keys in key_patterns.items():
        print(f"  {pattern}: {len(keys)} keys")
        if len(keys) <= 5:
            for key in keys:
                print(f"    - {key}")
        else:
            print(f"    - {keys[0]}")
            print(f"    - {keys[1]}")
            print(f"    - ... ({len(keys)-2} more)")
    
    # Look for transaction-like keys
    transaction_keys = []
    for key in all_keys:
        key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
        if 'transaction' in key_str.lower() or 'txn' in key_str.lower():
            transaction_keys.append(key_str)
    
    print(f"\n💳 Found {len(transaction_keys)} transaction-related keys")
    
    if transaction_keys:
        # Examine first few transaction keys
        print("\n🔍 Examining sample transactions:")
        for i, key in enumerate(transaction_keys[:3]):
            print(f"\n--- Transaction Key {i+1}: {key} ---")
            
            # Try different Redis data types
            try:
                key_type = r.type(key)
                if isinstance(key_type, bytes):
                    key_type = key_type.decode('utf-8')
                else:
                    key_type = str(key_type)
            except:
                key_type = 'unknown'
                
            print(f"Key type: {key_type}")
            
            try:
                if key_type == 'hash':
                    data = r.hgetall(key)
                    print("Hash data:")
                    for field, value in data.items():
                        field_str = field.decode('utf-8') if isinstance(field, bytes) else str(field)
                        value_str = value.decode('utf-8') if isinstance(value, bytes) else str(value)
                        print(f"  {field_str}: {value_str}")
                        
                elif key_type == 'string':
                    data = r.get(key)
                    data_str = data.decode('utf-8') if isinstance(data, bytes) else str(data)
                    print(f"String data: {data_str}")
                    
                    # Try to parse as JSON
                    try:
                        json_data = json.loads(data_str)
                        print("✅ Valid JSON structure:")
                        for k, v in json_data.items():
                            print(f"  {k}: {v}")
                    except:
                        print("❌ Not valid JSON")
                        
                elif key_type == 'list':
                    length = r.llen(key)
                    print(f"List length: {length}")
                    if length > 0:
                        sample = r.lrange(key, 0, 2)
                        for item in sample:
                            item_str = item.decode('utf-8') if isinstance(item, bytes) else str(item)
                            print(f"  List item: {item_str}")
                            
                elif key_type == 'set':
                    members = r.smembers(key)
                    print(f"Set members ({len(members)}):")
                    for member in list(members)[:5]:
                        member_str = member.decode('utf-8') if isinstance(member, bytes) else str(member)
                        print(f"  {member_str}")
                        
            except Exception as e:
                print(f"❌ Error reading key {key}: {e}")
    
    # Look for customer data
    print(f"\n👤 Looking for customer data...")
    customer_ids = set()
    
    # Try to find customer IDs from transaction keys
    for key in transaction_keys[:10]:  # Check first 10 keys
        try:
            key_type = r.type(key).decode('utf-8')
            
            if key_type == 'hash':
                customer_id = r.hget(key, 'customer_id')
                if customer_id:
                    customer_id = customer_id.decode('utf-8') if isinstance(customer_id, bytes) else customer_id
                    customer_ids.add(customer_id)
                    
            elif key_type == 'string':
                data = r.get(key)
                data_str = data.decode('utf-8') if isinstance(data, bytes) else str(data)
                try:
                    json_data = json.loads(data_str)
                    if 'customer_id' in json_data:
                        customer_ids.add(json_data['customer_id'])
                except:
                    pass
                    
        except Exception as e:
            continue
    
    print(f"Found {len(customer_ids)} unique customer IDs:")
    for customer_id in list(customer_ids)[:5]:
        print(f"  - {customer_id}")
    
    # Test the get_customer_transactions logic
    if customer_ids:
        print(f"\n🧪 Testing transaction retrieval for customer: {list(customer_ids)[0]}")
        test_customer = list(customer_ids)[0]
        
        # Simulate the logic from RiskScoringAgent
        pattern = "financial:transaction:*"  # Default pattern
        found_keys = r.keys(pattern)
        print(f"Keys matching pattern '{pattern}': {len(found_keys)}")
        
        # Try different patterns if nothing found
        if len(found_keys) == 0:
            patterns_to_try = [
                "*transaction*",
                "*:transaction:*", 
                "*financial*",
                "*TXN*",
                "*CUST*"
            ]
            
            for test_pattern in patterns_to_try:
                test_keys = r.keys(test_pattern)
                print(f"Keys matching pattern '{test_pattern}': {len(test_keys)}")
                if len(test_keys) > 0:
                    print(f"  Sample keys: {[k.decode('utf-8') if isinstance(k, bytes) else str(k) for k in test_keys[:3]]}")

def suggest_fixes():
    """Suggest potential fixes based on findings"""
    print(f"\n💡 POTENTIAL SOLUTIONS:")
    print("1. Check if transaction keys use a different pattern than 'financial:transaction:*'")
    print("2. Verify that customer_id field exists in transaction data")
    print("3. Ensure transaction data is stored as Redis hashes, not strings")
    print("4. Check if timestamp format is compatible with datetime.fromisoformat()")
    print("5. Verify numeric fields (amount, risk_score, lat, lon) can be converted to float")
    
    print(f"\n🔧 QUICK FIXES TO TRY:")
    print("1. Update the key pattern in redis_config.py")
    print("2. Modify the get_customer_transactions() method to match your Redis structure")
    print("3. Add error handling for data parsing issues")

if __name__ == "__main__":
    debug_redis_data()
    suggest_fixes()